import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import torch
# for debug
# from qwen_7b_chat.tokenization_qwen import QWenTokenizer as AutoTokenizer
# for realease
from transformers import AutoTokenizer
import tensorrt_llm
from tensorrt_llm.runtime import (
    ModelConfig, SamplingConfig, GenerationSession
)
from tensorrt_llm.runtime.generation import Mapping
from build import get_engine_name  # isort:skip
from default_config import default_config
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime.engine import Engine, get_engine_version

now_dir = os.path.dirname(os.path.abspath(__file__))


# copy from tensorrt_llm/runtime/generation.py to debug
class Qwen2ForCausalLMGenerationSession(GenerationSession):
    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer,
        mapping: Mapping,
        debug_mode=False,
        debug_tensors_to_save=None,
        cuda_graph_mode=False,
        stream: torch.cuda.Stream = None,
        global_max_input_length=default_config.max_input_len,
        global_max_output_length=default_config.max_output_len
    ):
        super().__init__(
            model_config,
            engine_buffer,
            mapping,
            debug_mode,
            debug_tensors_to_save=debug_tensors_to_save,
            cuda_graph_mode=cuda_graph_mode,
            stream=stream
        )
        self.global_max_input_length = global_max_input_length
        self.global_max_output_length = global_max_output_length

    def prepare_for_chat(
        self,
        pad_token_id: int,
        tokenizer,
        input_text: Union[str, List[str]],
        system_text: str = "You are a helpful assistant.",
        history: list = None,
        max_input_length: Union[int, None] = None,
    ):
        if max_input_length is None:
            max_input_length = self.global_max_input_length
        else:
            max_input_length = min(max_input_length, self.global_max_input_length)
        if history is None:
            history = []
        # pad_id = tokenizer.im_end_id
        # prepare for batch inference
        if not isinstance(input_text, list):
            batch_text = [input_text]
        else:
            batch_text = input_text
        if len(history) > 0 and len(history[0]) and len(history[0][0]) > 0 \
                and not isinstance(history[0][0], list):
            history_list = [history]
        elif len(history) == 0:
            history_list = [[]]
        else:
            history_list = history
        input_ids = []
        input_lengths = []

        for line, history in zip(batch_text, history_list):
            # use make_content to generate prompt
            # print("input_id_list len", len(input_id_list))
            messages = [
                {"role": "system", "content": system_text},
            ]
            for (query, response) in history:
                messages.append(
                    {"role": "user", "content": query}
                )
                messages.append(
                    {"role": "assistant", "content": response}
                )
            messages.append({"role": "user", "content": input_text})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_id = tokenizer([text], return_tensors="pt").input_ids
            input_id = input_id[-max_input_length:].type(torch.int32)
            input_ids.append(input_id)
            input_lengths.append(input_id.shape[-1])
        max_length = max(input_lengths)
        # do padding, should move outside the profiling to prevent the overhead
        for i in range(len(input_ids)):
            pad_size = max_length - input_lengths[i]

            pad = torch.ones([1, pad_size]).type(torch.int32) * pad_token_id
            input_ids[i] = torch.cat(
                [torch.IntTensor(input_ids[i]), pad], dim=-1)
        input_ids = torch.cat(input_ids, dim=0).cuda()
        input_lengths = torch.IntTensor(input_lengths).type(torch.int32).cuda()
        return input_ids, input_lengths
    
    def generate(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        sampling_config: SamplingConfig,
        max_new_tokens: int,
        runtime_rank: int = 0,
        stop_works_list: Optional[torch.Tensor] = None
    ):
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(
            max_new_tokens,
            self.global_max_output_length - max_input_length
        )
        # setup batch_size, max_input_length, max_output_len
        self.setup(
            batch_size=input_lengths.size(0),
            max_context_length=max_input_length,
            max_new_tokens=max_new_tokens
        )
        output_dict = self.decode(
            input_ids,
            input_lengths,
            sampling_config,
            output_sequence_lengths=True,
            return_dict=True,
            # stop_words_list=stop_works_list
        )
        with torch.no_grad():
            torch.cuda.synchronize()
            if runtime_rank == 0:
                output_dict['output_ids'] = output_dict['output_ids'][:, 0, :]
                return output_dict
    
    def chat(
        self,
        pad_token_id: int,
        tokenizer,
        sampling_config: SamplingConfig,
        input_text: Union[str, List[str]],
        system_text: str = "You are a helpful assistant.",
        history: list = None,
        max_input_length: Union[int, None] = None,
        max_new_tokens: Union[int, None] = None,
        runtime_rank: int = 0,
    ):
        input_ids, input_lengths = self.prepare_for_chat(
            pad_token_id=pad_token_id,
            tokenizer=tokenizer,
            input_text=input_text,
            system_text=system_text,
            history=history,
            max_input_length=max_input_length,
        )
        max_input_length = torch.max(input_lengths).item()
        if max_new_tokens is None:
            self.global_max_output_length
        else:
            max_new_tokens = min(
                max_new_tokens,
                self.global_max_output_length
            )
        max_input_length = torch.max(input_lengths).item()
        self.setup(
            batch_size=input_lengths.size(0),
            max_context_length=max_input_length,
            max_new_tokens=max_new_tokens
        )

        output_dict = self.decode(
            input_ids,
            input_lengths,
            sampling_config,
            streaming=False,
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        output_ids = output_dict['output_ids']
        sequence_lengths = output_dict['sequence_lengths']
        output_texts = []
        if runtime_rank == 0:
            for b in range(input_lengths.size(0)):
                inputs = input_ids[b]
                input_text = tokenizer.decode(inputs)
                # print(f'Input: \"{input_text}\"')
                outputs = output_ids[b][0, len(inputs): sequence_lengths[b][0]].tolist()
                output_text = tokenizer.decode(outputs, skip_special_tokens=False)
                output_texts.append(output_text)
        return output_texts

    def chat_stream(
        self,
        pad_token_id: int,
        stop_token_ids: List[int],
        tokenizer,
        sampling_config: SamplingConfig,
        input_text: Union[str, List[str]],
        system_text: str = "You are a helpful assistant.",
        history: list = None,
        max_input_length: Union[int, None] = None,
        max_new_tokens: Union[int, None] = None,
        runtime_rank: int = 0,
    ):
        input_ids, input_lengths = self.prepare_for_chat(
            pad_token_id=pad_token_id,
            tokenizer=tokenizer,
            input_text=input_text,
            system_text=system_text,
            history=history,
            max_input_length=max_input_length,
        )
        max_input_length = torch.max(input_lengths).item()
        # setup batch_size, max_input_length, max_output_len
        if max_new_tokens is None:
            max_new_tokens = self.global_max_output_length
        else:
            max_new_tokens = min(
                max_new_tokens,
                self.global_max_output_length
            )
        self.setup(
            batch_size=input_lengths.size(0),
            max_context_length=max_input_length,
            max_new_tokens=max_new_tokens
        )
        with torch.no_grad():
            chunk_lengths = input_lengths.clone()
            for output_dict in self.decode(
                input_ids,
                input_lengths,
                sampling_config,
                streaming=True,
                output_sequence_lengths=True,
                return_dict=True
            ):
                output_ids = output_dict['output_ids']
                sequence_lengths = output_dict['sequence_lengths']
                # print("sequence_lengths", sequence_lengths)
                torch.cuda.synchronize()
                if runtime_rank == 0:
                    output_texts = []
                    for i in range(output_ids.size(0)):
                        temp_ids = output_ids[i, 0, chunk_lengths[i]: sequence_lengths[i][0]]
                        pure_ids = []
                        for j in range(len(temp_ids)):
                            if temp_ids[j] in stop_token_ids:
                                pure_ids = temp_ids[:j]
                                break
                        if len(pure_ids) == 0:
                            pure_ids = temp_ids
                        if pure_ids.size(0) == 0:
                            continue
                        temp_text = tokenizer.decode(pure_ids, skip_special_tokens=False)
                        # check code is error
                        if b"\xef\xbf\xbd" in temp_text.encode():
                            continue
                        chunk_lengths[i] += pure_ids.size(0)
                        output_texts.append(temp_text)
                    if len(output_texts) > 0:
                        yield output_texts


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=default_config.engine_dir,
    )
    parser.add_argument(
        '--tokenizer_dir',
        type=str,
        default=default_config.tokenizer_dir,
        help="Directory containing the tokenizer.model."
    )
    default_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，请问你叫什么？<|im_end|>\n<|im_start|>assistant\n"
    parser.add_argument(
        '--input_text',
        type=str,
        # default='Born in north-east France, Soyer trained as a'
        default=default_text
    )
    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument(
        '--output_csv',
        type=str,
        help='CSV file where the tokenized output is stored.',
        default=None
    )
    parser.add_argument(
        '--output_npy',
        type=str,
        help='Numpy file where the tokenized output is stored.',
        default=None
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        help="Use beam search if num_beams >1",
        default=1
    )
    return parser.parse_args()


def get_model(tokenizer_dir, engine_dir, log_level='error', rank=0):
    # --load the tokenizer and engine #
    tensorrt_llm.logger.set_level(log_level)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        trust_remote_code=True,
    )
    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    gen_config_path = os.path.join(tokenizer_dir, 'generation_config.json')
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    top_p = gen_config['top_p']
    pad_token_id = eos_token_id = gen_config["eos_token_id"][0]
    stop_token_ids = gen_config["eos_token_id"]
    # new load way
    engine_version = get_engine_version(engine_dir)
    assert engine_version is not None
    engine = Engine.from_dir(engine_dir, rank)
    pretrained_config = engine.config.pretrained_config
    build_config = engine.config.build_config

    tp_size = pretrained_config.mapping.tp_size
    num_heads = pretrained_config.num_attention_heads // tp_size
    num_kv_heads = pretrained_config.num_key_value_heads
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    hidden_size = pretrained_config.hidden_size // tp_size
    head_size = pretrained_config.head_size
    model_config = ModelConfig(
        max_batch_size=build_config.max_batch_size,
        vocab_size=pretrained_config.vocab_size,
        num_layers=pretrained_config.num_hidden_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        gpt_attention_plugin=bool(build_config.plugin_config.gpt_attention_plugin),
        remove_input_padding=bool(build_config.plugin_config.remove_input_padding),
        paged_kv_cache=bool(build_config.plugin_config.paged_kv_cache),
        tokens_per_block=build_config.plugin_config.tokens_per_block,
        quant_mode=pretrained_config.quant_mode,
        # gather_context_logits=build_config.gather_context_logits,
        # gather_generation_logits=build_config.gather_generation_logits,
        dtype=pretrained_config.dtype,
        use_custom_all_reduce=build_config.plugin_config.use_custom_all_reduce,
    )
    sampling_config = SamplingConfig(
        end_id=eos_token_id,
        pad_id=pad_token_id,
        num_beams=1,
        top_p=top_p,
        length_penalty=1,
        repetition_penalty=1.1,
        min_length=0,
    )
    runtime_mapping = pretrained_config.mapping
    return (
        engine, model_config, sampling_config, runtime_mapping,
        tokenizer, eos_token_id, pad_token_id, stop_token_ids
    )


def generate(
    max_new_tokens: int,
    log_level: str = 'error',
    engine_dir: str = 'qwen_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    tokenizer_dir: str = None,
    num_beams: int = 1,
):

    runtime_rank = tensorrt_llm.mpi_rank()
    (

        engine, model_config, sampling_config, runtime_mapping,
        tokenizer, eos_token_id, pad_token_id, stop_token_ids
    ) = get_model(tokenizer_dir, engine_dir, log_level, rank=runtime_rank)
    engine_buffer = engine.engine
    build_config = engine.config.build_config
    decoder = Qwen2ForCausalLMGenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping,
    )
    input_tokens = []
    input_tokens.append(
        tokenizer.encode(input_text, add_special_tokens=False))

    input_ids = torch.tensor(input_tokens, device="cuda", dtype=torch.int32)
    input_lengths = torch.tensor(
        [input_ids.size(1)], device="cuda", dtype=torch.int32
    )
    max_input_length = torch.max(input_lengths).item()
    max_new_tokens = min(
        max_new_tokens,
        build_config.max_output_len
    )
    decoder.setup(
        batch_size=input_lengths.size(0),
        max_context_length=max_input_length,
        max_new_tokens=max_new_tokens
    )

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    if runtime_rank == 0:
        for b in range(input_lengths.size(0)):
            inputs = input_tokens[b]
            input_text = tokenizer.decode(inputs)
            print(f'Input: \"{input_text}\"')
            if num_beams <= 1:
                outputs = output_ids[b][0, len(inputs): ].tolist()
                output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                print(f'Output: \"{output_text}\"')
            else:
                for beam in range(num_beams):
                    outputs = output_ids[b][beam, len(inputs): ].tolist()
                    output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                    print(f'Output(beam: {beam}): \"{output_text}\"')

        output_ids = output_ids.reshape((-1, output_ids.size(2)))
    return


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
