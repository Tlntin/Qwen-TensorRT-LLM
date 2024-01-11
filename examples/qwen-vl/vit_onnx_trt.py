import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torchvision import transforms
from transformers import AutoConfig
from typing import List
from torchvision.transforms import InterpolationMode
from PIL import Image
import requests
import os
import tensorrt as trt
import argparse

from tensorrt_llm._utils import str_dtype_to_torch

import tensorrt as trt
from itertools import tee

from polygraphy.backend.trt import (
    network_from_onnx_path,
    engine_from_network,
    save_engine,
    Profile,
)

from polygraphy.backend.trt import CreateConfig
from tensorrt import MemoryPoolType

class Preprocss:
    def __init__(self,
                 image_size:int,
                 ):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size,image_size),
                interpolation = InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
            
        ])
        
    def encode(self,image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path,stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return images
    
class ONNX_TRT:
    def __init__(self,image_size):
        self.image_size = image_size
    def export_onnx(self,onnx_file_path,pretrained_model_path):
        
        image_pre_obj = Preprocss(self.image_size)
        torch_dtype = str_dtype_to_torch("float32")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="cpu",
            torch_dtype=torch_dtype,
            fp32=True,
            trust_remote_code=True
        ).eval()
        image_url = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg']
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        image = image_pre_obj.encode(image_url).to(device)
        if not os.path.exists('./input_pt'):
            os.mkdir('./input_pt')
            torch.save(image, './input_pt/image.pt')
        #model_visual = model.transformer.visual.to(device).to(torch_dtype)
        model_visual = model.transformer.visual
        model_visual.eval()
        
        torch.onnx.export(model_visual,
                        image.to('cuda'),
                        onnx_file_path,
                        opset_version=17,
                        input_names=['input'],
                        output_names = ['output'],
                        dynamic_axes = {
                            'input':{0:'batch'}
                        }
                        )
    def generate_trt_engine(self,onnxFile,planFile,use_polygraph,minBS=1,optBS=2,maxBS=4):
        import tensorrt as trt
        from time import time

        ## There are two ways to convert an engine
        ## 1. the first is to use the polygraph tool, which can use fp16;
        ## 2. the second is to use the native trt api, which must use fp32, if use fp16 the accuracy loss is great
        ## 
        ## todo: the difference between the two ways!!
        if use_polygraph:
            print("we are using polygraph tools get engine file !!!")
            #preview_features = [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
            preview_features = []
            
            profiles = [Profile().add(
                "input",
                min=(minBS, 3, self.image_size, self.image_size ),
                opt=(optBS, 3, self.image_size, self.image_size ), # Optimized based on the inputs.
                max=(maxBS, 3, self.image_size, self.image_size ),
            )]
            trt_inference_config = CreateConfig(
                            fp16=True,
                            memory_pool_limits = {MemoryPoolType.WORKSPACE: 2048 * 1024 * 1024},
                            profiles=profiles,
                            precision_constraints=("obey"),
                            builder_optimization_level=3,
                            preview_features=preview_features
                        )
            
            onnx_network = network_from_onnx_path(onnxFile)
            
            trt_engine = engine_from_network(onnx_network, trt_inference_config)
            
            save_engine(trt_engine, planFile)

        else:
            print("we are using tensorrt api get engine file !!!")
            logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            # breakpoint()
            #config.set_flag(trt.BuilderFlag.FP16)
            #config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

            parser = trt.OnnxParser(network, logger)
            print("======onnxFile",onnxFile)

            with open(onnxFile, 'rb') as model:
                if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                    print("Failed parsing %s" % onnxFile)
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                print("Succeeded parsing %s" % onnxFile)
            print("Begin convert onnx to TensorRT engine, need wait a few minutes")

            nBS = -1
            nMinBS = minBS
            nOptBS = optBS
            nMaxBS = maxBS
            inputT = network.get_input(0)
            inputT.shape = [nBS, 3, self.image_size, self.image_size]
            profile.set_shape(inputT.name, [nMinBS, 3, self.image_size, self.image_size],
                              [nOptBS, 3, self.image_size, self.image_size], [nMaxBS, 3, self.image_size, self.image_size])

            config.add_optimization_profile(profile)

            t0 = time()
            engineString = builder.build_serialized_network(network, config)
            t1 = time()
            if engineString == None:
                print("Failed building %s" % planFile)
            else:
                print("Succeeded building %s in %d s" % (planFile, t1 - t0))
            print("plan file is",planFile)
            with open(planFile, 'wb') as f:
                f.write(engineString)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnxFile',type=str, default='./onnx/visual_encoder/visual_encoder.onnx',help='')#onnx/visual_encoder
    parser.add_argument('--pretrained_model_path',type=str, default='./Qwen-VL-Chat',help='')
    parser.add_argument('--planFile',type=str, default='./plan/visual_encoder/visual_encoder_fp16.plan',help='')
    parser.add_argument('--only_trt', action='store_true', help='Run only convert the onnx to TRT engine.')
    parser.add_argument('--minBS',type=int, default=1)
    parser.add_argument('--optBS',type=int, default=1)
    parser.add_argument('--maxBS',type=int, default=4)
    parser.add_argument('--use_polygraph', action='store_true', help='if use polygraph tools get engine.')
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    
    args = parse_arguments()
    onnx_file_dir = os.path.dirname(args.onnxFile)
    if not os.path.exists(onnx_file_dir):
        os.makedirs(onnx_file_dir)
    plan_file_dir = os.path.dirname(args.planFile)
    if not os.path.exists(plan_file_dir):
        os.makedirs(plan_file_dir)
    if True:
        onnx_trt_obj = ONNX_TRT(448)
    else:
        onnx_trt_obj = ONNX_TRT(config.visual['image_size'])
    
    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile,args.planFile,args.minBS,args.optBS,args.maxBS,args.use_polygraph)
    else:
        onnx_trt_obj.export_onnx(args.onnxFile,args.pretrained_model_path)
        onnx_trt_obj.generate_trt_engine(args.onnxFile,args.planFile,args.use_polygraph,args.minBS,args.optBS,args.maxBS)
        
    
        



