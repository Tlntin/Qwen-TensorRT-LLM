#!/usr/bin/env bash

set -ex

pushd examples/gpt

python3 build.py --log_level=error --dtype=float32 --n_layer=2 --random_seed=16879570956913565 --output_dir=gpt_engine_fp32
python3 build.py --log_level=error --dtype=float32 --n_layer=2 --random_seed=16879570956913565 --use_gpt_attention_plugin=float32 --output_dir=gpt_engine_fp32_plugin
python3 build.py --log_level=error --dtype=float16 --n_layer=2 --random_seed=16879570956913565 --output_dir=gpt_engine_fp16
python3 build.py --log_level=error --dtype=float16 --n_layer=2 --random_seed=16879570956913565 --use_gpt_attention_plugin=float16 --output_dir=gpt_engine_fp16_plugin
python3 build.py --log_level=error --dtype=float16 --n_layer=2 --random_seed=16879570956913565 --use_gpt_attention_plugin=float16 --remove_input_padding --output_dir=gpt_engine_fp16_plugin_ragged

popd
