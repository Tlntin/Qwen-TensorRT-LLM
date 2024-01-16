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
from utils.utils import make_context
from default_config import default_config
from tensorrt_llm.quantization import QuantMode

now_dir = os.path.dirname(os.path.abspath(__file__))


# copy from tensorrt_llm/runtime/generation.py to debug
class QWenForCausalLMGenerationSession(GenerationSession):
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
        global_max_output_length=default_config.max_input_len + default_config.max_new_tokens,
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
        pad_id = tokenizer.im_end_id
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
            _, input_id_list = make_context(
                tokenizer=tokenizer,
                query=line,
                history=history,
                system=system_text,
                max_input_length=max_input_length,
            )
            # print("input_id_list len", len(input_id_list))
            input_id = torch.from_numpy(
                np.array(input_id_list, dtype=np.int32)
            ).type(torch.int32).unsqueeze(0)
            input_ids.append(input_id)
            input_lengths.append(input_id.shape[-1])
        max_length = max(input_lengths)
        # do padding, should move outside the profiling to prevent the overhead
        for i in range(len(input_ids)):
            pad_size = max_length - input_lengths[i]

            pad = torch.ones([1, pad_size]).type(torch.int32) * pad_id
            input_ids[i] = torch.cat(
                [torch.IntTensor(input_ids[i]), pad], axis=-1)
        input_ids = torch.cat(input_ids, axis=0).cuda()
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
            tokenizer=tokenizer,
            input_text=input_text,
            system_text=system_text,
            history=history,
            max_input_length=max_input_length,
        )
        max_input_length = torch.max(input_lengths).item()
        if max_new_tokens is None:
            max_new_tokens = self.global_max_output_length - max_input_length
        else:
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
            streaming=False,
            output_sequence_lengths=True,
            return_dict=True,
        )
        with torch.no_grad():
            torch.cuda.synchronize()
            output_ids = output_dict['output_ids']
            sequence_lengths = output_dict['sequence_lengths']
            if runtime_rank == 0:
                output_texts = [
                    tokenizer.decode(
                        output_ids[i, 0, input_lengths[i]: sequence_lengths[i][0]],
                        skip_special_tokens=False
                    )
                    for i in range(output_ids.size(0))
                ]
                return output_texts

    def chat_stream(
        self,
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
            tokenizer=tokenizer,
            input_text=input_text,
            system_text=system_text,
            history=history,
            max_input_length=max_input_length,
        )
        max_input_length = torch.max(input_lengths).item()
        # setup batch_size, max_input_length, max_output_len
        if max_new_tokens is None:
            max_new_tokens = self.global_max_output_length - max_input_length
        else:
            max_new_tokens = min(
                max_new_tokens,
                self.global_max_output_length - max_input_length
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
                            if temp_ids[j] in [tokenizer.im_start_id, tokenizer.im_end_id]:
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


def get_model(tokenizer_dir, engine_dir, log_level='error'):
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
    top_k = gen_config['top_k']
    top_p = gen_config['top_p']
    chat_format = gen_config['chat_format']
    if chat_format == "raw":
        eos_token_id = gen_config['eos_token_id']
        pad_token_id = gen_config['pad_token_id']
    elif chat_format == "chatml":
        pad_token_id = eos_token_id = tokenizer.im_end_id
    else:
        raise Exception("unkown chat format ", chat_format)

    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    #num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size=world_size, rank=runtime_rank, tp_size=tp_size, pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        remove_input_padding=remove_input_padding,
        dtype=dtype,
        quant_mode=quant_mode,
        use_custom_all_reduce=use_custom_all_reduce
    )
    sampling_config = SamplingConfig(
        end_id=eos_token_id,
        pad_id=pad_token_id,
        num_beams=1,
        top_k = top_k,
        top_p = top_p,
        length_penalty=1,
        repetition_penalty=1.1,
        min_length=0,
    )

    engine_name = get_engine_name('qwen', dtype, tp_size, pp_size, runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)
    print(f'Loading engine from {serialize_path}')
    return (
        model_config, sampling_config, runtime_mapping, runtime_rank,
        serialize_path, remove_input_padding, 
        tokenizer, eos_token_id, pad_token_id
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
    (
        model_config, sampling_config, runtime_mapping, runtime_rank,
        serialize_path, remove_input_padding, 
        tokenizer, eos_token_id, pad_token_id
    ) = get_model(tokenizer_dir, engine_dir, log_level)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = QWenForCausalLMGenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping,
    )

    input_tokens = []
    if input_file is None:
        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_tokens.append(np.array(line, dtype='int32'))
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                row = row[row != eos_token_id]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = None
    if input_file is None:
        input_ids = torch.tensor(input_tokens, device="cuda", dtype=torch.int32)
        input_lengths = torch.tensor(
            [input_ids.size(1)], device="cuda", dtype=torch.int32
        )
    else:
        input_lengths = torch.tensor(
            [len(x) for x in input_tokens],
            device="cuda",
            dtype=torch.int32
        )
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(
                input_ids, device="cuda", dtype=torch.int32
            ).unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                eos_token_id).cuda()

    max_input_length = torch.max(input_lengths).item()
    max_new_tokens = min(
        max_new_tokens,
        default_config.max_input_len + default_config.max_new_tokens - max_input_length
    )
    decoder.setup(
        batch_size=input_lengths.size(0),
        max_context_length=max_input_length,
        max_new_tokens=max_new_tokens
    )

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    if runtime_rank == 0:
        if output_csv is None and output_npy is None:
            for b in range(input_lengths.size(0)):
                inputs = input_tokens[b]
                input_text = tokenizer.decode(inputs)
                print(f'Input: \"{input_text}\"')
                if num_beams <= 1:
                    # outputs = output_ids[b][0].tolist()
                    # output_text = _decode_chatml(
                    #     outputs,
                    #     stop_words=[],
                    #     eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
                    #     tokenizer=tokenizer,
                    #     raw_text_len=len(input_text),
                    #     context_length=len(inputs)
                    # )
                    outputs = output_ids[b][0, len(inputs): ].tolist()
                    output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                    print(f'Output: \"{output_text}\"')
                else:
                    for beam in range(num_beams):
                        # outputs = output_ids[b][beam].tolist()
                        # output_text = _decode_chatml(
                        #     outputs,
                        #     stop_words=[],
                        #     eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
                        #     tokenizer=tokenizer,
                        #     raw_text_len=len(input_text),
                        #     context_length=len(inputs)
                        # )
                        outputs = output_ids[b][beam, len(inputs): ].tolist()
                        output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                        print(f'Output(beam: {beam}): \"{output_text}\"')

        output_ids = output_ids.reshape((-1, output_ids.size(2)))

        if output_csv is not None:
            output_file = Path(output_csv)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            outputs = output_ids.tolist()
            with open(output_file, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerows(outputs)

        if output_npy is not None:
            output_file = Path(output_npy)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
            np.save(output_file, outputs)
    return


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
