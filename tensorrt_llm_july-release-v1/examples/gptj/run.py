import argparse
import json
import os

import torch
from utils import token_encoder

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "merges.txt"
VOCAB_FILE = "vocab.json"

PAD_ID = 50256
START_ID = 50256
END_ID = 50256


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='gpt_outputs')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument(
        '--input_tokens',
        type=str,
        help='CSV file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument(
        '--hf_model_location',
        type=str,
        default="./",
        help=
        'The hugging face model location stores the merges.txt and vocab.json to create tokenizer'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    max_input_len = config['builder_config']['max_input_len']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('gptj', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    vocab_file = os.path.join(args.hf_model_location, VOCAB_FILE)
    merges_file = os.path.join(args.hf_model_location, MERGES_FILE)
    assert os.path.exists(vocab_file), f"{vocab_file} does not exist"
    assert os.path.exists(merges_file), f"{merges_file} does not exist"
    encoder = token_encoder.get_encoder(vocab_file, merges_file)
    input_ids = None
    input_text = None
    input_lengths = []
    if args.input_tokens is None:
        input_text = args.input_text
        input_ids = torch.IntTensor(encoder.encode(
            args.input_text)).cuda().unsqueeze(0)
        input_lengths = torch.cuda.IntTensor([input_ids.size(1)])
    else:
        input_ids = []
        with open(args.input_tokens) as f_in:
            i = 0
            for line in f_in:
                input_ids.append([])
                for e in line.strip().split(','):
                    input_ids[i].append(int(e))
                this_len = len(input_ids[i])
                input_lengths.append(this_len)
                i += 1
        input_text = "<ids from file>"
        # Create tensors which pad shorter tensors.
        max_batch_seqlen = min(max(input_lengths), max_input_len)
        input_ids_tensor = torch.ones(
            (len(input_ids), max_batch_seqlen), dtype=torch.int32) * PAD_ID
        for i, seq in enumerate(input_ids):
            seqlen = min(max_batch_seqlen, len(seq))
            input_ids_tensor[i, :seqlen] = torch.IntTensor(seq)[:seqlen]
        input_ids = input_ids_tensor.cuda()
    input_lengths = torch.tensor(input_lengths).int().cuda()

    model_config = ModelConfig(num_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding)
    sampling_config = SamplingConfig(end_id=END_ID,
                                     pad_id=PAD_ID,
                                     num_beams=args.num_beams,
                                     min_length=args.min_length)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)
    if remove_input_padding:
        decoder.setup(1, torch.max(input_lengths).item(), args.max_output_len)
    else:
        decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len)
    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    if args.num_beams > 1:
        # For beam search, the output is next to the input.
        # Create a output tensor and copy the id over
        output_copy = torch.ones(len(input_ids),
                                 args.max_output_len,
                                 args.num_beams,
                                 dtype=torch.int32) * END_ID
        input_lengths = input_lengths.cpu().numpy()
        for i in range(len(input_ids)):
            start = input_lengths[i]
            out_len = output_ids.shape[1] - start
            out_len = min(out_len, args.max_output_len)
            output_copy[i, :out_len, :] = output_ids[i,
                                                     start:start + out_len, :]
        output_ids = output_copy.cuda()
        # Select the index from beams based on the best cum_log_probs
        top_idx = torch.argmax(decoder.cum_log_probs, dim=1)
        output_ids = output_ids[range(output_ids.shape[0]), :,
                                top_idx].squeeze(-1)
    else:
        output_ids = output_ids[:, len(input_ids[0]):]

    output_text = encoder.batch_decode(output_ids.tolist())
    print(f'Input: {input_text}')
    print(f'Output: {output_text}')
