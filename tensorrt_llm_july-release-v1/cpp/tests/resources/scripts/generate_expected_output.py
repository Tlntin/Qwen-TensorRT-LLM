#!/usr/bin/env python3

from pathlib import Path

import run
import run_hf


def generate_output(engine: str,
                    num_beams: int,
                    output_name: str,
                    max_output_len: int = 8):

    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    engine_dir = models_dir / 'rt_engine/gpt2' / engine / '1-gpu/'

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens.npy'
    if num_beams <= 1:
        output_dir = data_dir / 'sampling'
    else:
        output_dir = data_dir / ('beam_search_' + str(num_beams))

    run.generate(engine_dir=str(engine_dir),
                 input_file=str(input_file),
                 output_npy=str(output_dir / (output_name + '.npy')),
                 output_csv=str(output_dir / (output_name + '.csv')),
                 max_output_len=max_output_len,
                 num_beams=num_beams)


def generate_outputs(num_beams):
    generate_output(engine='fp32-default',
                    num_beams=num_beams,
                    output_name='output_tokens_fp32')
    generate_output(engine='fp32-plugin',
                    num_beams=num_beams,
                    output_name='output_tokens_fp32_plugin')
    generate_output(engine='fp16-default',
                    num_beams=num_beams,
                    output_name='output_tokens_fp16')
    generate_output(engine='fp16-plugin',
                    num_beams=num_beams,
                    output_name='output_tokens_fp16_plugin')
    generate_output(engine='fp16-plugin-packed',
                    num_beams=num_beams,
                    output_name='output_tokens_fp16_plugin_packed')


def generate_hf_output(data_type: str,
                       output_name: str,
                       max_output_len: int = 8):

    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_dir = models_dir / 'gpt2'

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens.npy'
    output_dir = data_dir / 'huggingface'

    run_hf.generate(model_dir=str(model_dir),
                    data_type=data_type,
                    input_file=str(input_file),
                    output_npy=str(output_dir / (output_name + '.npy')),
                    output_csv=str(output_dir / (output_name + '.csv')),
                    max_output_len=max_output_len)


def generate_hf_outputs():
    generate_hf_output(data_type='fp32',
                       output_name='output_tokens_fp32_huggingface')
    generate_hf_output(data_type='fp16',
                       output_name='output_tokens_fp16_huggingface')


if __name__ == '__main__':
    generate_outputs(num_beams=1)
    generate_outputs(num_beams=2)
    generate_hf_outputs()
