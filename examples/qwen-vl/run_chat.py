import torch
from run import QWenInfer, Vit, parse_arguments


if __name__ == '__main__':
    args = parse_arguments()
    # stream = torch.cuda.current_stream().cuda_stream
    # vit = Vit(args.vit_engine_dir, args.log_level)
    # image_embeds = vit.run(args.input_dir, stream)
    # qinfer = QWenInfer(args.tokenizer_dir,args.qwen_engine_dir,args.log_level,args.output_csv,args.output_npy,args.num_beams)
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
        print('\n')
        
        # content_list = args.images_path
        content_list = [{'text': input_text}]

        if len(history) == 0:
            query = qinfer.tokenizer.from_list_format(content_list)
        else:
            query = input_text
        
        response = ""
        for new_text in qinfer.qwen_infer_stream(
            input_text=query,
            max_new_tokens=args.max_new_tokens,
            history=history
        ):
            print(new_text, end='', flush=True)
            response += new_text
        print("")
        history.append((input_text, response))
    
    
    
    
    
