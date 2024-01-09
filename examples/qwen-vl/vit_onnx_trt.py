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
        torch_dtype = str_dtype_to_torch("float16")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="cpu",
            torch_dtype=torch_dtype,
            # fp16=True,
            trust_remote_code=True
        ).eval()
        image_url = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg']
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        image = image_pre_obj.encode(image_url).to(device)
        if not os.path.exists('./input_pt'):
            os.mkdir('./input_pt')
            torch.save(image, './input_pt/image.pt')
        model_visual = model.transformer.visual.to(device).to(torch_dtype)
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
    def generate_trt_engine(self,onnxFile,planFile,minBS=1,optBS=2,maxBS=4):
        import tensorrt as trt
        from time import time
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

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
    parser.add_argument('--pretrained_model_path',type=str, default='./qwen/Qwen-VL-Chat',help='')
    parser.add_argument('--planFile',type=str, default='./plan/visual_encoder/visual_encoder_fp16.plan',help='')
    parser.add_argument('--only_trt', action='store_true', help='Run only convert the onnx to TRT engine.')
    parser.add_argument('--minBS',type=int, default=1)
    parser.add_argument('--optBS',type=int, default=1)
    parser.add_argument('--maxBS',type=int, default=4)
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
        onnx_trt_obj.generate_trt_engine(args.onnxFile,args.planFile,args.minBS,args.optBS,args.maxBS)
    else:
        onnx_trt_obj.export_onnx(args.onnxFile,args.pretrained_model_path)
        onnx_trt_obj.generate_trt_engine(args.onnxFile,args.planFile,args.minBS,args.optBS,args.maxBS)
        
    
        



