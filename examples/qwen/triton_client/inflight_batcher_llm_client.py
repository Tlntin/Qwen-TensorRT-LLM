# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import queue
import sys
import time
from functools import partial
import os
import sys
import json
import numpy as np
from transformers import PreTrainedTokenizer
from typing import List, Tuple
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


now_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(now_dir)
sys.path.append(parent_dir)

from utils.utils import make_context
#
# Simple streaming client for TRT-LLM inflight bacthing backend
#
# In order for this code to work properly, config.pbtxt must contain these values:
#
# model_transaction_policy {
#   decoupled: True
# }

# parameters: {
#   key: "gpt_model_type"
#   value: {
#     string_value: "inflight_batching"
#   }
# }
#
# In order for gpt_model_type 'inflight_batching' to work, you must copy engine from
#
# tensorrt_llm/cpp/tests/resources/models/rt_engine/gpt2/fp16-inflight-batching-plugin/1-gpu/
#


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_inputs(
        input_ids_data,
        input_lengths_data,
        request_output_len_data,
        end_id_data,
        pad_id_data,
        beam_width_data,
        temperature_data,
        streaming_data = None
):
    protocol = 'grpc'
    inputs = [
        prepare_tensor("input_ids", input_ids_data, protocol),
        prepare_tensor("input_lengths", input_lengths_data, protocol),
        prepare_tensor("request_output_len", request_output_len_data,
                       protocol),
        prepare_tensor("end_id", end_id_data, protocol),
        prepare_tensor("pad_id", pad_id_data, protocol),
        prepare_tensor("beam_width", beam_width_data, protocol),
        prepare_tensor("temperature", temperature_data, protocol),
    ]
    if streaming_data is not None:
        inputs.append(prepare_tensor("streaming", streaming_data, protocol))

    return inputs


def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('input_ids', [1, 1], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "UINT32"),
        grpcclient.InferInput('stop', [1, 1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
    inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.uint32))
    inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))

    return inputs


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(tokenizer, response, user_data, result, error):
    if error:
        # print("error", error)
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        if (FLAGS.streaming):
            output_ids = result.as_numpy('output_ids')
            tokens = list(output_ids[0][0])
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            # print(tokens, flush=True)
            print(text, flush=True, end="")
            response["response"] += text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL encrypted channel to the server",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "-r",
        "--root-certificates",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded root certificates. Default is None.",
    )
    parser.add_argument(
        "-p",
        "--private-key",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded private key. Default is None.",
    )
    parser.add_argument(
        "-x",
        "--certificate-chain",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded certificate chain. Default is None.",
    )
    parser.add_argument(
        "-C",
        "--grpc-compression-algorithm",
        type=str,
        required=False,
        default=None,
        help=
        "The compression algorithm to be used when sending request to server. Default is None.",
    )
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=True,
        help="Enable streaming mode. Default is False.",
    )
    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )
    parser.add_argument(
        "--request-output-len",
        type=int,
        required=False,
        default=512,
        help="temperature value",
    )
    parser.add_argument(
        '--stop-after-ms',
        type=int,
        required=False,
        default=0,
        help='Early stop the generation after a few milliseconds')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        required=False,
                        default=os.path.join(parent_dir, "qwen_7b_chat"),
                        help='Specify tokenizer directory')
    parser.add_argument('--tokenizer_type',
                        type=str,
                        default='auto',
                        required=False,
                        choices=['auto', 't5', 'llama'],
                        help='Specify tokenizer type')
    parser.add_argument('--request_id',
                        type=str,
                        default='1',
                        required=False,
                        help='The request_id for the stop request')

    FLAGS = parser.parse_args()
    history1 = []
    print("欢迎使用Qwen聊天机器人，输入exit退出，输入clear清空历史记录")
    print('=' * 20)
    while True:
        query = input("Human: ")
        if query == 'exit' or query == "exit()":
            break
        if query == 'clear':
            history1 = []
            continue
        if FLAGS.tokenizer_type == 't5':
            tokenizer = T5Tokenizer(vocab_file=FLAGS.tokenizer_dir,
                                    padding_side='left')
        elif FLAGS.tokenizer_type == 'auto':
            tokenizer = AutoTokenizer.from_pretrained(
                FLAGS.tokenizer_dir,
                padding_side='left',
                trust_remote_code=True,
            )
        elif FLAGS.tokenizer_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                       legacy=False,
                                                       padding_side='left')
        else:
            raise AttributeError(
                f'Unexpected tokenizer type: {FLAGS.tokenizer_type}')
        # tokenizer.pad_token = tokenizer.eos_token
        # pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
        # end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
        gen_config_path = os.path.join(FLAGS.tokenizer_dir, 'generation_config.json')
        with open(gen_config_path, 'r') as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == "raw":
            eos_id = gen_config['eos_token_id']
            pad_id = gen_config['pad_token_id']
        elif chat_format == "chatml":
            pad_id = eos_id = tokenizer.im_end_id
        else:
            raise Exception("unkown chat format ", chat_format)
        eos_token = tokenizer.decode(eos_id)
        tokenizer.eos_token = eos_token
        tokenizer.pad_token = eos_token
        raw_text, input_id_list = make_context(
            tokenizer=tokenizer,
            query=query,
            history=history1,
        )
        # print("raw_text", raw_text)
        # input_ids = [tokenizer.encode(FLAGS.text)]
        input_ids = [input_id_list]
        # print("input_lenght", len(input_ids[0]))
        input_ids_data = np.array(input_ids, dtype=np.int32)
        input_lengths = [[len(ii)] for ii in input_ids]
        input_lengths_data = np.array(input_lengths, dtype=np.int32)
        request_output_len = [[FLAGS.request_output_len]]
        request_output_len_data = np.array(request_output_len, dtype=np.int32)
        eos_ids = [[eos_id]]
        end_id_data = np.array(eos_ids, dtype=np.int32)
        pad_ids = [[pad_id]]
        pad_id_data = np.array(pad_ids, dtype=np.int32)
        beam_width = [[FLAGS.beam_width]]
        beam_width_data = np.array(beam_width, dtype=np.int32)
        temperature = [[FLAGS.temperature]]
        temperature_data = np.array(temperature, dtype=np.float32)
        if FLAGS.streaming:
            streaming = [[FLAGS.streaming]]
            streaming_data = np.array(streaming, dtype=bool)
        else:
            streaming_data = None

        inputs = prepare_inputs(
            input_ids_data,
            input_lengths_data,
            request_output_len_data,
            end_id_data,
            pad_id_data,
            beam_width_data,
            temperature_data,
            streaming_data,
        )

        if FLAGS.stop_after_ms > 0:
            stop_inputs = prepare_stop_signals()
        else:
            stop_inputs = None

        request_id = FLAGS.request_id

        if FLAGS.streaming:
            actual_output_ids = [input_ids[0]]
        else:
            actual_output_ids = []

        user_data = UserData()
        # print(f'Input: {query}')
        print("Output: ", end="", flush=True)
        response = {"response": ""}
        with grpcclient.InferenceServerClient(
                url=FLAGS.url,
                verbose=FLAGS.verbose,
                ssl=FLAGS.ssl,
                root_certificates=FLAGS.root_certificates,
                private_key=FLAGS.private_key,
                certificate_chain=FLAGS.certificate_chain,
        ) as triton_client:
            try:

                if FLAGS.streaming:

                    # Establish stream
                    triton_client.start_stream(
                        callback=partial(callback, tokenizer, response, user_data),
                        stream_timeout=FLAGS.stream_timeout,
                    )
                    # Send request
                    triton_client.async_stream_infer(
                        'tensorrt_llm',
                        inputs,
                        request_id=request_id,
                    )

                    if stop_inputs is not None:

                        time.sleep(FLAGS.stop_after_ms / 1000.0)

                        triton_client.async_stream_infer(
                            'tensorrt_llm',
                            stop_inputs,
                            request_id=request_id,
                            parameters={'Streaming': FLAGS.streaming})

                    #Wait for server to close the stream
                    triton_client.stop_stream()

                    # Parse the responses
                    while True:
                        try:
                            result = user_data._completed_requests.get(block=False)
                        except Exception as err:
                            # print("Debug get error", err)
                            break

                        if type(result) == InferenceServerException:
                            print("Received an error from server:")
                            print(result)
                        else:
                            output_ids = result.as_numpy('output_ids')
                            if output_ids is not None:
                                if (FLAGS.streaming):
                                    # Only one beam is supported
                                    tokens = list(output_ids[0][0])
                                    # text = tokenizer.decode(tokens, skip_special_tokens=True)
                                    # print(text, flush=True, end="")
                                    actual_output_ids[
                                        0] = actual_output_ids[0] + tokens
                                else:
                                    for beam_output_ids in output_ids[0]:
                                        tokens = list(beam_output_ids)
                                        actual_output_ids.append(tokens)
                            else:
                                print("Got cancellation response from server")
                else:
                    # Send request
                    triton_client.async_infer(
                        'tensorrt_llm',
                        inputs,
                        request_id=request_id,
                        callback=partial(callback, tokenizer, response, user_data),
                        parameters={'Streaming': FLAGS.streaming})

                    if stop_inputs is not None:

                        time.sleep(FLAGS.stop_after_ms / 1000.0)

                        triton_client.async_infer(
                            'tensorrt_llm',
                            stop_inputs,
                            request_id=request_id,
                            callback=partial(callback, tokenizer, response, user_data),
                            parameters={'Streaming': FLAGS.streaming})

                    processed_count = 0
                    expected_responses = 1 + (1 if stop_inputs is not None else 0)
                    while processed_count < expected_responses:
                        try:
                            result = user_data._completed_requests.get()
                            print("Got completed request", flush=True)
                        except Exception:
                            break

                        if type(result) == InferenceServerException:
                            print("Received an error from server:")
                            print(result)
                        else:
                            output_ids = result.as_numpy('output_ids')
                            if output_ids is not None:
                                for beam_output_ids in output_ids[0]:
                                    tokens = list(beam_output_ids)
                                    actual_output_ids.append(tokens)
                            else:
                                print("Got response for cancellation request")

                        processed_count = processed_count + 1
            except Exception as e:
                print("channel creation failed: " + str(e))
                sys.exit()

            passed = True

            # print("output_ids = ", actual_output_ids)
            if not FLAGS.streaming:
                output_ids = np.array(actual_output_ids)
                output_ids = output_ids.reshape(
                    (output_ids.size, )).tolist()[input_ids_data.shape[1]:]
                response = tokenizer.decode(output_ids, skip_special_tokens=True)
                print(response)
            else:
                print("")
            history1.append((query, response["response"]))
            # sys.exit(not passed)
