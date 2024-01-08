import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Union
import tensorrt as trt
import torch
from transformers import AutoTokenizer,AutoConfig, PreTrainedTokenizer
import numpy as np

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import Session, TensorInfo
from tensorrt_llm.runtime import (
    ModelConfig, SamplingConfig, GenerationSession
)
from tensorrt_llm.runtime.generation import Mapping
from build import get_engine_name  # isort:skip

from tensorrt_llm.quantization import QuantMode
import csv
from run import  QWenInfer, vit_process, parse_arguments



if __name__ == '__main__':
    args = parse_arguments()

    stream = torch.cuda.current_stream().cuda_stream
    
    image_embeds = vit_process(args.input_dir,args.vit_engine_dir,args.log_level,stream)
    qinfer= QWenInfer(args.tokenizer_dir,args.qwen_engine_dir,args.log_level,args.output_csv,args.output_npy,args.num_beams)
    qinfer.qwen_model_init()
    
    run_i = 0
    history = []
    while True:
        input_text = None
        try:
            input_text = input("Text (or 'q' to quit): ")
        except:
            continue
            
        if input_text == "clear history":
            history = []
            continue

        
        if input_text.lower() == 'q':
            break
        print('\n')
        
        content_list = args.images_path
        content_list.append({'text': input_text})

        if run_i == 0:
            query = qinfer.tokenizer.from_list_format(content_list)
        else:
            query = input_text
        
        run_i = run_i + 1
        
        qinfer.qwen_infer(image_embeds,None,query,args.max_new_tokens,history)
    
    
    
    
    
