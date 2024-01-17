import torch
from run import QWenInfer, Vit, parse_arguments
from vit_onnx_trt import Preprocss


if __name__ == '__main__':
    args = parse_arguments()
    # load vit with custom image
    """
    image_preprocess = Preprocss(image_size=448)
    image_paths = ["demo.jpeg"]
    images = image_preprocess.encode(image_paths)
    image_paths = [{"image": image} for image in image_paths]
    vit = Vit(args.vit_engine_dir, args.log_level)
    input_vit = vit.run(images=images)
    """
    # otherwise
    input_vit = None
    image_paths = []
    qinfer = QWenInfer(args.tokenizer_dir,args.qwen_engine_dir, args.log_level)
    qinfer.qwen_model_init()

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

        # content_list = args.images_path
        if len(history) == 0:
            content_list = image_paths + [{'text': input_text}]
            query = qinfer.tokenizer.from_list_format(content_list)
        else:
            query = input_text
        
        response = ""
        for new_text in qinfer.qwen_infer_stream(
            input_vit=input_vit,
            input_text=query,
            max_new_tokens=args.max_new_tokens,
            history=history
        ):
            print(new_text, end='', flush=True)
            response += new_text
        print("")
        history.append((input_text, response))
    
    
    
    
    
