import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, GenerationConfig
# for debug
# from qwen_7b_chat.tokenization_qwen import QWenTokenizer as AutoTokenizer
# for release
from transformers import AutoTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from run import QWenForCausalLMGenerationSession
from utils import make_context

from build import get_engine_name  # isort:skip


now_dir = os.path.dirname(os.path.abspath(__file__))


def TRT_QWen(args, config):
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    world_size = config['builder_config']['tensor_parallel']
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    multi_query_mode = config['builder_config']['multi_query_mode']

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        multi_query_mode=multi_query_mode,
        remove_input_padding=remove_input_padding
    )

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('qwen', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tensorrt_llm.logger.set_level(args.log_level)

    profiler.start('load tensorrt_llm engine')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = QWenForCausalLMGenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping
    )
    profiler.stop('load tensorrt_llm engine')
    tensorrt_llm.logger.info(
        f'Load engine takes: {profiler.elapsed_time_in_sec("load tensorrt_llm engine")} sec'
    )
    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location
    profiler.start('load tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_location,
        legacy=False,
        padding_side='left',
        trust_remote_code=True,
    )
    profiler.stop('load tokenizer')
    tensorrt_llm.logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )
    tokenizer.pad_token = tokenizer.eos_token
    dataset_cnn = load_dataset(
        "ccdv/cnn_dailymail",
        '3.0.0'
    )
    gen_config_path = os.path.join(hf_model_location, 'generation_config.json')
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    chat_format = gen_config['chat_format']

    max_batch_size = args.batch_size

    # runtime parameters
    # repetition_penalty = 1
    top_k = args.top_k
    output_len = 100
    test_token_num = 923
    # top_p = 0.0
    # random_seed = 5
    temperature = 1
    num_beams = args.num_beams

    # pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    # end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
    tokenizer.pad_token_id = pad_id = end_id = tokenizer.im_end_id
    # use this prompt to make chat model do summarize
    system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."

    if test_trt_llm:
        config_path = os.path.join(args.engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        tensorrt_llm_qwen = TRT_QWen(args, config)

    if test_hf:
        profiler.start('load HF model')
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_location,
            device_map='auto',
            trust_remote_code=True,
        )
        model.generation_config = GenerationConfig.from_pretrained(
            hf_model_location,
            trust_remote_code=True
        )
        profiler.stop('load HF model')
        tensorrt_llm.logger.info(
            f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec'
        )
        if args.data_type == 'fp16':
            model.half()
        model.cuda()

    def summarize_tensorrt_llm(datapoint):
        batch_size = len(datapoint['article'])
        assert batch_size > 0
        line = copy.copy(datapoint['article'])
        line_encoded = []
        input_lengths = []
        for i in range(batch_size):
            line[i] = line[i] + ' TL;DR: '

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")
            # use make_content to generate prompt
            _, input_id_list = make_context(
                tokenizer=tokenizer,
                query=line[i],
                history=[],
                system=system_prompt,
            )
            input_id = torch.from_numpy(
                np.array(input_id_list, dtype=np.int32)
            ).type(torch.int32).unsqueeze(0)
            input_id = input_id[:, -test_token_num:]

            line_encoded.append(input_id)
            input_lengths.append(input_id.shape[-1])

        # do padding, should move outside the profiling to prevent the overhead
        max_length = max(input_lengths)
        if tensorrt_llm_qwen.remove_input_padding:
            line_encoded = [torch.IntTensor(t).cuda() for t in line_encoded]
        else:
            # do padding, should move outside the profiling to prevent the overhead
            for i in range(batch_size):
                pad_size = max_length - input_lengths[i]

                pad = torch.ones([1, pad_size]).type(torch.int32) * pad_id
                line_encoded[i] = torch.cat(
                    [torch.IntTensor(line_encoded[i]), pad], axis=-1)

            line_encoded = torch.cat(line_encoded, axis=0).cuda()
            input_lengths = torch.IntTensor(input_lengths).type(
                torch.int32).cuda()

        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=end_id, pad_id=pad_id, top_k=top_k, num_beams=num_beams)

        with torch.no_grad():
            tensorrt_llm_qwen.setup(batch_size,
                                     max_input_length=max_length,
                                     max_new_tokens=output_len)

            if tensorrt_llm_qwen.remove_input_padding:
                output_ids, end_step = tensorrt_llm_qwen.decode_batch(
                    line_encoded, sampling_config)
            else:
                output_ids, end_step = tensorrt_llm_qwen.decode(
                    line_encoded,
                    input_lengths,
                    sampling_config,
                )

            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        output_beams_list = [
            tokenizer.batch_decode(
                output_ids[
                    batch_idx,
                    :,
                    input_lengths[batch_idx]:input_lengths[batch_idx] + end_step
                ],
                skip_special_tokens=True
            )
            for batch_idx in range(batch_size)
        ]
        return (
            output_beams_list,
            output_ids[:, :, max_length: max_length + end_step].tolist()
        )
    def get_stop_words_ids(chat_format, tokenizer):
        if chat_format == "raw":
            stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
        elif chat_format == "chatml":
            stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")
        return stop_words_ids

    def summarize_hf(datapoint):
        batch_size = len(datapoint['article'])
        assert batch_size > 0
        if batch_size > 1:
            logger.warning(
                f"HF does not support batch_size > 1 to verify correctness due to padding. Current batch size is {batch_size}"
            )

        line = copy.copy(datapoint['article'])
        
        new_line_list = []
        if batch_size > 1:
            for i in range(batch_size):
                line[i] = line[i] + ' TL;DR: '

                line[i] = line[i].strip()
                line[i] = line[i].replace(" n't", "n't")
                # use make_content to generate prompt
                raw_text, _ = make_context(
                    tokenizer=tokenizer,
                    query=line[i],
                    history=[],
                    system=system_prompt,
                    chat_format=chat_format
                )
                new_line_list.append(raw_text)
            line_encoded = tokenizer(
                new_line_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )["input_ids"].type(torch.int64)
        else:
            line[0] = line[0] + ' TL;DR: '
            line[0] = line[0].strip()
            line[0] = line[0].replace(" n't", "n't")
            # use make_content to generate prompt
            _, input_id_list = make_context(
                tokenizer=tokenizer,
                query=line[0],
                history=[],
                system=system_prompt,
                chat_format=chat_format
            )
            line_encoded = torch.from_numpy(
                np.array(input_id_list, dtype=np.int32)
            ).type(torch.int32).unsqueeze(0)

        line_encoded = line_encoded[:, -test_token_num:]
        line_encoded = line_encoded.cuda()

        stop_words_ids=[]
        stop_words_ids.extend(get_stop_words_ids(
            chat_format, tokenizer
        ))

        with torch.no_grad():
            output = model.generate(
                line_encoded,
                max_new_tokens=len(line_encoded[0]) + output_len,
                top_k=top_k,
                temperature=temperature,
                # eos_token_id=tokenizer.im_end_id,
                # pad_token_id=tokenizer.im_end_id,
                stop_words_ids=stop_words_ids,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True
            )

        tokens_list = output[:, len(line_encoded[0]): len(line_encoded[0]) + output_len].tolist()
        output = output.reshape([batch_size, num_beams, -1])
        output_lines_list = [
            tokenizer.batch_decode(
                output[:, i, len(line_encoded[0]): len(line_encoded[0]) + output_len],
                skip_special_tokens=True
            )
            for i in range(num_beams)
        ]

        return output_lines_list, tokens_list

    if test_trt_llm:
        datapoint = dataset_cnn['test'][0:1]
        summary, _ = summarize_tensorrt_llm(datapoint)
        if runtime_rank == 0:
            logger.info(
                "---------------------------------------------------------")
            logger.info("TensorRT-LLM Generated : ")
            logger.info(f" Article : {datapoint['article']}")
            logger.info(f"\n Highlights : {datapoint['highlights']}")
            logger.info(f"\n Summary : {summary}")
            logger.info(
                "---------------------------------------------------------")

    if test_hf:
        datapoint = dataset_cnn['test'][0:1]
        summary, _ = summarize_hf(datapoint)
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Article : {datapoint['article']}")
        logger.info(f"\n Highlights : {datapoint['highlights']}")
        logger.info(f"\n Summary : {summary}")
        logger.info("---------------------------------------------------------")

    metric_tensorrt_llm = [load("rouge") for _ in range(num_beams)]
    metric_hf = [load("rouge") for _ in range(num_beams)]
    for i in range(num_beams):
        metric_tensorrt_llm[i].seed = 0
        metric_hf[i].seed = 0

    ite_count = 0
    data_point_idx = 0
    while (data_point_idx < len(dataset_cnn['test'])) and (ite_count <
                                                           args.max_ite):
        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        datapoint = dataset_cnn['test'][data_point_idx:(data_point_idx +
                                                        max_batch_size)]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            summary_tensorrt_llm, tokens_tensorrt_llm = summarize_tensorrt_llm(
                datapoint)
            profiler.stop('tensorrt_llm')

        if test_hf:
            profiler.start('hf')
            summary_hf, tokens_hf = summarize_hf(datapoint)
            profiler.stop('hf')

        if runtime_rank == 0:
            if test_trt_llm:
                for batch_idx in range(len(summary_tensorrt_llm)):
                    for beam_idx in range(num_beams):
                        metric_tensorrt_llm[beam_idx].add_batch(
                            predictions=[
                                summary_tensorrt_llm[batch_idx][beam_idx]
                            ],
                            references=[datapoint['highlights'][batch_idx]])
            if test_hf:
                for beam_idx in range(num_beams):
                    for batch_idx in range(len(summary_hf[beam_idx])):
                        metric_hf[beam_idx].add_batch(
                            predictions=[summary_hf[beam_idx][batch_idx]],
                            references=[datapoint['highlights'][batch_idx]])

            logger.debug('-' * 100)
            logger.debug(f"Article : {datapoint['article']}")
            if test_trt_llm:
                logger.debug(f'TensorRT-LLM Summary: {summary_tensorrt_llm}')
            if test_hf:
                logger.debug(f'HF Summary: {summary_hf}')
            logger.debug(f"highlights : {datapoint['highlights']}")

        data_point_idx += max_batch_size
        ite_count += 1

    if runtime_rank == 0:
        if test_trt_llm:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
            )
            for beam_idx in range(num_beams):
                logger.info(f"TensorRT-LLM beam {beam_idx} result")
                computed_metrics_tensorrt_llm = metric_tensorrt_llm[
                    beam_idx].compute()
                for key in computed_metrics_tensorrt_llm.keys():
                    logger.info(
                        f'  {key} : {computed_metrics_tensorrt_llm[key] * 100}'
                    )

                if args.check_accuracy and beam_idx == 0:
                    assert computed_metrics_tensorrt_llm['rouge1'] * 100 > args.tensorrt_llm_rouge1_threshold
        if test_hf:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'Hugging Face (total latency: {profiler.elapsed_time_in_sec("hf")} sec)'
            )
            for beam_idx in range(num_beams):
                logger.info(f"HF beam {beam_idx} result")
                computed_metrics_hf = metric_hf[beam_idx].compute()
                for key in computed_metrics_hf.keys():
                    logger.info(
                        f'{key} : {computed_metrics_hf[key] * 100}'
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hf_model_location',
        type=str,
        default=os.path.join(now_dir, "qwen_7b_chat")
    )
    parser.add_argument(
        '--test_hf',
        action='store_true',
        # default=True,
    )
    parser.add_argument(
        '--test_trt_llm',
        action='store_true',
        # default=True,
    )
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp32', 'fp16'],
                        default='fp16')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=""
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=os.path.join(now_dir, "trt_engines", "fp16", "1-gpu")
    )
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)

    args = parser.parse_args()

    main(args)
