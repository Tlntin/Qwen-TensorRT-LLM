"""Benchmark offline inference throughput."""
import argparse
import json
import os
import random
import time
from typing import List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
)
from tqdm import tqdm, trange
from run_old import get_model, Qwen2ForCausalLMGenerationSession
from default_config import default_config
import tensorrt_llm


now_dir = os.path.dirname(os.path.abspath(__file__))


def sample_requests(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    num_requests: int,
    chat_format: str = "chatml",
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    max_input_len = default_config.max_input_len
    max_output_len = default_config.max_input_len + default_config.max_new_tokens
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    # completions = [completion for _, completion in dataset]
    # completion_token_ids = 
    tokenized_dataset = []
    for i in trange(500, desc="Tokenizing for sample"):
        prompt = dataset[i][0]
        output_text = dataset[i][1]
        if chat_format == 'chatml':
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            raw_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif chat_format == 'raw':
            raw_text = prompt
        prompt_tokens = tokenizer([raw_text], return_tensors="pt").input_ids.squeeze()
        new_token_len = len(tokenizer(output_text).input_ids)
        tokenized_dataset.append((raw_text, prompt_tokens, new_token_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, new_token_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or new_token_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > max_input_len or (prompt_len + new_token_len) > max_output_len:
            # Prune too long sequences.
            continue
        # limit by max_output_len 
        filtered_dataset.append((prompt, prompt_len, new_token_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def run_trt_llm(
    requests: List[Tuple[str, int, int]],
    engine_dir: str,
    tokenizer_dir: str,
    n: int,
    max_batch_size: int,
    log_level = "error",
) -> float:
    global_max_input_len = default_config.max_input_len
    global_max_output_len = default_config.max_input_len + default_config.max_new_tokens
    if max_batch_size > default_config.trt_max_batch_size:
        raise Exception(
            "max batch size {} must be lower than trt_max_batch_size {}".format(
                max_batch_size, default_config.trt_max_batch_size
            )
        )
    runtime_rank = tensorrt_llm.mpi_rank()
    (

        engine, model_config, sampling_config, runtime_mapping,
        tokenizer, eos_token_id, pad_token_id, stop_token_ids
    ) = get_model(tokenizer_dir, engine_dir, log_level, rank=runtime_rank)

    engine_buffer = engine.engine
    decoder = Qwen2ForCausalLMGenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping,
    )

    # Add the requests to the engine.
    sampling_config.num_beams = n
    sampling_config.temperature = 0.0 if n > 1 else default_config.temperature
    sampling_config.top_p = default_config.top_p
    sampling_config.top_k = default_config.top_k
    start = time.time()

    batch: List[str] = []
    max_new_tokens = 0
    total_num_tokens = []
    last_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    stop_token_ids = [last_token_id, pad_token_id]
    for i, (prompt, prompt_len, new_token_len) in tqdm(enumerate(requests), total=len(requests)):
        # Add the prompt to the batch. 
        batch.append(prompt)
        max_new_tokens = max(max_new_tokens, new_token_len)
        if len(batch) < max_batch_size and i < len(requests) - 1:
            continue
        input_ids = []
        input_lengths = []
        for input_text in batch:
            input_id = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=global_max_input_len,
            ).input_ids.type(torch.int32)
            input_ids.append(input_id)
            input_lengths.append(input_id.shape[-1])
        # padding
        max_length = max(input_lengths)
        # do padding, should move outside the profiling to prevent the overhead
        for i in range(len(input_ids)):
            pad_size = max_length - input_lengths[i]
            pad = torch.ones([1, pad_size]).type(torch.int32) * pad_token_id
            input_ids[i] = torch.cat(
                [torch.IntTensor(input_ids[i]), pad],
                dim=-1
            )
        # do inference
        input_ids = torch.cat(input_ids, dim=0).cuda()
        input_lengths = torch.IntTensor(input_lengths).type(torch.int32).cuda()
        output_dict = decoder.generate(
            input_ids=input_ids,
            input_lengths=input_lengths,
            sampling_config=sampling_config,
            max_new_tokens=min(max_new_tokens, global_max_output_len - input_ids.shape[1]),
        )
        output_ids = output_dict['output_ids']
        sequence_lengths = output_dict['sequence_lengths']
        pure_output_ids = []
        for i in range(len(batch)):
            temp_ids = output_ids[i, input_lengths[i]: sequence_lengths[i][0]]
            pure_ids = []
            for i in range(len(temp_ids)):
                if temp_ids[i] in stop_token_ids:
                    pure_ids = temp_ids[:i + 1]
                    break
            if len(pure_ids) == 0:
                pure_ids = temp_ids
            pure_output_ids.append(pure_ids)
        # get the output text
        output_texts = [
            tokenizer.decode(out_ids, skip_special_tokens=True)
            for out_ids in pure_output_ids
        ]
        # get the total num of tokens
        output_lengths = [len(out_ids) for out_ids in pure_output_ids]
        assert len(output_lengths) == len(batch)
        for input_len, new_token_len in zip(input_lengths, output_lengths):
            total_num_tokens.append(input_len + new_token_len)
        batch = []
        max_new_tokens = 0

    end = time.time()
    during = end - start
    sum_total_num_tokens = sum(total_num_tokens)
    return during, sum_total_num_tokens


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    # use_beam_search: bool,
    max_batch_size: int,
    chat_format: str = "chatml",
) -> float:
    global_max_input_len = default_config.max_input_len
    global_max_output_len = default_config.max_input_len + default_config.max_new_tokens
    # assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    elif llm.config.model_type == "qwen":
        tokenizer.pad_token = tokenizer.decode(tokenizer.im_end_id)
    llm = llm.cuda()
    # stop_words_ids = []
    # stop_words_ids.extend(get_stop_words_ids(
    #         chat_format, tokenizer
    # ))
    # stop_words_ids2 = [idx for ids in stop_words_ids for idx in ids]
    pbar = tqdm(total=len(requests))
    start = time.time()
    total_num_tokens = []
    batch: List[str] = []
    input_lengths: List[int] = []
    max_prompt_len = 0
    max_new_tokens = 0
    for i in range(len(requests)):
        prompt, prompt_len, new_token_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        input_lengths.append(prompt_len)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_new_tokens = max(max_new_tokens, new_token_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            temp_input_max = max(max_prompt_len, next_prompt_len)
            temp_new_token_max = max(max_new_tokens, next_output_len)
            if temp_input_max <= global_max_input_len and \
                (temp_input_max + temp_new_token_max) <= global_max_output_len:
                continue
        # Generate the sequences.
        input_ids = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=global_max_input_len,
        ).input_ids

        # limit the max_new_tokens
        max_new_tokens = min(max_new_tokens, global_max_output_len - input_ids.shape[1])
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            # stop_words_ids=stop_words_ids,
            num_return_sequences=n,
            top_k=default_config.top_k,
            top_p=default_config.top_p,
            temperature=default_config.temperature,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
        pure_output_ids = llm_outputs[:, input_ids.shape[-1]:]
        # get the output text
        output_texts = tokenizer.batch_decode(
            pure_output_ids, skip_special_tokens=True
        )
        output_lengths = []
        for out_ids in pure_output_ids:
            early_stop = False
            for i in range(len(out_ids)):
                if out_ids[i] in llm.generation_config.eos_token_id:
                    output_lengths.append(i + 1)
                    early_stop = True
                    break
            if not early_stop:
                output_lengths.append(len(out_ids))
        assert len(output_lengths) == len(batch)
        for input_len, new_token_len in zip(input_lengths, output_lengths):
            total_num_tokens.append(input_len + new_token_len)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        input_lengths = []
        max_prompt_len = 0
        max_new_tokens = 0
    end = time.time()
    during = end - start
    sum_total_num_tokens = sum(total_num_tokens)
    return during, sum_total_num_tokens


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        padding_side='left',
        trust_remote_code=True,
    )
    requests = sample_requests(
        tokenizer=tokenizer,
        dataset_path=args.dataset,
        num_requests=args.num_prompts,
        chat_format=args.chat_format
    )

    if args.backend == "trt_llm":
        elapsed_time, total_num_tokens = run_trt_llm(
            requests=requests,
            engine_dir=args.engine_dir,
            tokenizer_dir=args.tokenizer_dir,
            n=args.n,
            max_batch_size=args.trt_max_batch_size,
        )
    elif args.backend == "hf":
        # assert args.tensor_parallel_size == 1
        elapsed_time, total_num_tokens = run_hf(
            requests=requests,
            model=args.hf_model_dir,
            tokenizer=tokenizer,
            n=args.n,
            # use_beam_search=args.use_beam_search,
            max_batch_size=args.hf_max_batch_size,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    # total_num_tokens = sum(
    #     prompt_len + output_len
    #     for _, prompt_len, output_len in requests
    # )
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["trt_llm", "hf"],
        default="trt_llm",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default=os.path.join(
            now_dir,
            "ShareGPT_V3_unfiltered_cleaned_split.json"
        ),
        help="Path to the dataset."
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default=default_config.hf_model_dir,
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=default_config.tokenizer_dir,
    )
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=default_config.engine_dir,
    )
    # parser.add_argument(
    #     "--tensor-parallel-size",
    #     "-tp",
    #     type=int,
    #     default=1
    # )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt."
    )

    # parser.add_argument(
    #     "--use-beam-search",
    #     action="store_true"
    # )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--hf_max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size for HF backend."
    )

    parser.add_argument(
        "--trt_max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size for TRT-LLM backend."
    )
    parser.add_argument(
        "--chat-format",
        type=str,
        default=default_config.chat_format,
        choices=["chatml", "raw"],
        help="choice the model format, base or chat"
    )
    args = parser.parse_args()

    if args.backend == "trt-llm":
        if args.trt_max_batch_size is None:
            raise ValueError("trt max batch size is requried for TRT-LLM backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("hf max batch size is required for HF backend.")
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model

    main(args)
