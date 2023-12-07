# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re

import torch
import transformers

import tensorrt_llm
from tensorrt_llm import runtime
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop", "<|system|>", "<|user|>", "<|assistant|>",
                          "<|observation|>"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='trtModel')
    parser.add_argument('--input_text', type=str, default='续写：北京市教育资源丰富')
    parser.add_argument(
        '--input_tokens',
        type=str,
        help='CSV file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.8)
    return parser.parse_args()


def process_response(responseList):
    for i, response in enumerate(responseList):
        response = response.strip()
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0],
                              r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0],
                              r"%s\1" % item[1], response)

        responseList[i] = response
    return responseList

def build_single_message(role, metadata, message):
    tokenizer = SPTokenizer('/data/zhaohb/chatglm3-6b-32k/339f17ff464d47b5077527c2b34e80a7719ede3e/tokenizer.model')
    assert role in ["system", "user", "assistant", "observation"], role
    #get_command(f"<|{'user'}|>") = 64795
    role_tokens = [64795] + tokenizer.encode(f"{metadata}\n")
    message_tokens = tokenizer.encode(message)
    tokens = role_tokens + message_tokens
    return tokens

def build_chat_input(tokenizer, query, history=None, role="user"):
    if history is None:
        history = []
    input_ids = []
    input_ids.extend(build_single_message(role, "", query))
    #get_command("<|assistant|>") = 64796
    input_ids.extend([64796])
    from transformers import PreTrainedTokenizer
    return tokenizer.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)['input_ids'].int().contiguous().cuda()

if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('chatglm2-6b', dtype, world_size,
                                  runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/data/zhaohb/chatglm3-6b-32k/339f17ff464d47b5077527c2b34e80a7719ede3e/", trust_remote_code=True)
    input_ids = None
    input_text = None
    if args.input_tokens is None:
        input_text = args.input_text
        #input_ids = build_inputs(tokenizer=tokenizer, query=input_text, history=[])
        input_ids = build_chat_input(tokenizer=tokenizer, query=input_text, history=[])
        print(input_ids)
        #import numpy as np
        #input_ids = np.array([[64790, 64792, 64795, 30910,    13, 30910, 54993, 55172, 31211, 33693, 48679, 32115, 64796]])
        #input_ids = torch.from_numpy(input_ids).int().contiguous().cuda()
        #print(input_ids)
    else:
        input_ids = []
        with open(args.input_tokens) as f_in:
            for line in f_in:
                for e in line.strip().split(','):
                    input_ids.append(int(e))
        input_text = "<ids from file>"
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int32).cuda().unsqueeze(0)
    input_lengths = torch.tensor(
        [input_ids.size(1) for _ in range(input_ids.size(0))]).int().cuda()

    model_config = ModelConfig(model_name="chatglm6b",
                               num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               dtype=dtype)
    sampling_config = SamplingConfig(end_id=2,
                                     pad_id=0,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = runtime.GenerationSession(model_config, engine_buffer,
                                        runtime_mapping, debug_mode=True)
    decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len)

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    # [output_len, batch_size, beam_width] -> [batch_size, output_len, beam_width]
    output_ids = output_ids.squeeze(1)
    torch.cuda.synchronize()
    for i in range(len(output_ids.tolist())):
        output_ids = output_ids.tolist()[i][input_ids.size(1):]

        pure_ids = []
        for i in range(len(output_ids)):
            #if output_ids[i] in [tokenizer.eos_token_id, tokenizer.bos_token_id]:
            if output_ids[i] in [1, 2]:
                pure_ids = output_ids[:i]
                break
        if len(pure_ids) == 0:
            pure_ids = output_ids

        outputList = tokenizer.batch_decode(pure_ids,
                                            skip_special_tokens=True)
        output_text = process_response(outputList)
        print(f'***************************************')
        print(f'Input --->\n {input_text}')
        print(f'Output --->\n {"".join(output_text)}')
        print(f'***************************************')

    print("Finished!")
