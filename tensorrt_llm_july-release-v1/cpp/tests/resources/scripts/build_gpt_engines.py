#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

import hf_gpt_convert as _egc
import torch.multiprocessing as multiprocessing

import build as _egb  # isort:skip


def build_engine(weigth_dir: Path, engine_dir: Path, *args):
    _egb.run_build([
        '--model_dir',
        str(weigth_dir),
        '--output_dir',
        str(engine_dir),
        '--log_level=error',
        '--max_batch_size=256',
        '--max_input_len=40',
        '--max_output_len=20',
        '--max_beam_width=2',
        '--builder_opt=0',
    ] + list(args))


def build_engines():
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    gpt2_dir = models_dir / 'gpt2'

    print("Pulling gpt2 from huggingface")
    subprocess.check_call(["rm", "-rf", str(gpt2_dir)])
    subprocess.check_call(
        ["git", "clone", "https://huggingface.co/gpt2",
         str(gpt2_dir)])
    pytroch_model = str(gpt2_dir / "pytorch_model.bin")
    subprocess.check_call(["rm", pytroch_model])
    subprocess.check_call([
        "wget", "-q",
        "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin", "-O",
        pytroch_model
    ])

    # set env for engine building
    os.environ['__LUNOWUD'] = '-autotune:num_cask_tactics=1'

    weight_dir = models_dir / 'c-model/gpt2'
    engine_dir = models_dir / 'rt_engine/gpt2'

    print("\nConverting to fp32")
    fp32_weight_dir = weight_dir / 'fp32/1-gpu'
    _egc.run_conversion(
        _egc.ProgArgs(in_file=str(gpt2_dir),
                      out_dir=str(fp32_weight_dir),
                      storage_type='float32'))

    print("\nBuilding fp32 engines")
    fp32_weight_dir_1_gpu = fp32_weight_dir / '1-gpu'
    build_engine(fp32_weight_dir_1_gpu, engine_dir / 'fp32-default/1-gpu',
                 '--dtype=float32')
    build_engine(fp32_weight_dir_1_gpu, engine_dir / 'fp32-plugin/1-gpu',
                 '--dtype=float32', '--use_gpt_attention_plugin=float32')

    print("\nConverting to fp16")
    fp16_weight_dir = weight_dir / 'fp16/1-gpu'
    _egc.run_conversion(
        _egc.ProgArgs(in_file=str(gpt2_dir),
                      out_dir=str(fp16_weight_dir),
                      storage_type='float16'))

    print("\nBuilding fp16 engines")
    fp16_weight_dir_1_gpu = fp16_weight_dir / '1-gpu'
    build_engine(fp16_weight_dir_1_gpu, engine_dir / 'fp16-default/1-gpu',
                 '--dtype=float16')
    build_engine(fp16_weight_dir_1_gpu, engine_dir / 'fp16-plugin/1-gpu',
                 '--dtype=float16', '--use_gpt_attention_plugin=float16')
    build_engine(fp16_weight_dir_1_gpu, engine_dir / 'fp16-plugin-packed/1-gpu',
                 '--dtype=float16', '--use_gpt_attention_plugin=float16',
                 '--remove_input_padding')
    build_engine(fp16_weight_dir_1_gpu,
                 engine_dir / 'fp16-inflight-batching-plugin/1-gpu',
                 '--dtype=float16', '--use_inflight_batching=float16',
                 '--remove_input_padding')

    print("Done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    build_engines()
