# Guide to QWen-VL pipeline
1. Download Qwen-VL-Chat
    ```bash
    git lfs install
    git clone https://huggingface.co/Qwen/Qwen-VL-Chat
    ```
2. ViT
- Generate ONNX model and TRT engine for ViT
    ```bash
    python vit_onnx_trt.py --pretrained_model_path ./Qwen-VL-Chat
    ```
    The exported ONNX files lies in `./onnx/visual_encoder` and the built engine lie in `./plan/visual_encoder`. And you have onnx files already and convert TRT engine only, use:
    ```bash
    python vit_onnx_trt.py --pretrained_model_path ./Qwen-VL-Chat --only_trt
    ```
    Moreover, it will save test image tensor to `image.pt` and visual query tokens to `query_tokens.pt` for later pipeline inference.

3. QwenVL(fp16)

- Build TRT-LLM engines (only need to add --max_prompt_embedding_table_size)

    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    ```bash
    python3 build.py  \
	--hf_model_dir=./Qwen-VL-Chat \
	--dtype float16 --max_batch_size 4 \
	--max_input_len 512 --max_new_tokens 1024 \
	--remove_input_padding \
	--use_gpt_attention_plugin float16 \
	--use_gemm_plugin float16 --enable_context_fmha \
	--use_rmsnorm_plugin --log_level error \
	--use_lookup_plugin float16 \
	--max_prompt_embedding_table_size 2048 \
	--output_dir=trt_engines/Qwen-VL-7B-fp16
    ```
    The built Qwen engines lie in `./trt_engines/Qwen-VL-7B-fp16`.

4. Qwen-VL(int8 weight only) 
    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    ```bash
    python3 build.py  \
	--hf_model_dir=./Qwen-VL-Chat \
	--dtype float16 --max_batch_size 4 \
	--max_input_len 512 --max_new_tokens 1024 \
	--remove_input_padding \
	--use_gpt_attention_plugin float16 \
	--use_gemm_plugin float16 --enable_context_fmha \
	--use_rmsnorm_plugin --log_level error \
	--use_lookup_plugin float16 \
	--max_prompt_embedding_table_size 2048 \
        --use_weight_only --weight_only_precision int8 \
	--output_dir=trt_engines/Qwen-VL-7B-int8
    ```
    - The built Qwen engines lie in `./trt_engines/Qwen-VL-7B-int8`.

5. Qwen-VL(int4 weight only) 
    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    ```bash
    python3 build.py  \
	--hf_model_dir=./Qwen-VL-Chat \
	--dtype float16 --max_batch_size 4 \
	--max_input_len 512 --max_new_tokens 1024 \
	--remove_input_padding \
	--use_gpt_attention_plugin float16 \
	--use_gemm_plugin float16 --enable_context_fmha \
	--use_rmsnorm_plugin --log_level error \
	--use_lookup_plugin float16 \
	--max_prompt_embedding_table_size 2048 \
        --use_weight_only --weight_only_precision int4 \
	--output_dir=trt_engines/Qwen-VL-7B-int4
    ```
    - The built Qwen engines lie in `./trt_engines/Qwen-VL-7B-int4`.

6. Qwen-VL(gptq-int4)
    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    - install some python package
    ```bash
    pip install auto-gptq optimum
    pip install transformers -U
    ```
   
    - convert int4-gptq weight
    ```bash
    python3 gptq_convert.py --hf_model_dir ./Qwen-VL-Chat --tokenizer_dir ./Qwen-VL-Chat --quant_ckpt_path ./Qwen-VL-Chat-My-Int4
    ```
   
    - build engine
   ```bash
   python3 build.py \
	--hf_model_dir=./Qwen-VL-Chat \
	--dtype float16 --max_batch_size 4 \
	--max_input_len 512 --max_new_tokens 1024 \
	--remove_input_padding \
	--use_gpt_attention_plugin float16 \
	--use_gemm_plugin float16 --enable_context_fmha \
	--use_rmsnorm_plugin --log_level error \
	--use_lookup_plugin float16 \
	--max_prompt_embedding_table_size 2048 \
	--use_weight_only \
        --weight_only_precision int4_gptq \
        --per_group \
        --quant_ckpt_path ./Qwen-VL-Chat-My-Int4/gptq_model-4bit-128g.safetensors \
	--output_dir=trt_engines/Qwen-VL-7B-int4-gptq 
   ```

7. Run Qwen-VL pipeline
    - fp16 run
    ```bash
    python run.py \
	--tokenizer_dir=./Qwen-VL-Chat \
	--qwen_engine_dir=./trt_engines/Qwen-VL-7B-fp16/ \
	--vit_engine_dir=./plan/
    ```
   
    - int8 weight only run
    ```bash
    python run.py \
         --tokenizer_dir=./Qwen-VL-Chat \
         --qwen_engine_dir=trt_engines/Qwen-VL-7B-int8 \
         --vit_engine_dir=./plan/
    ```
   
    - int4 weight only run
    ```bash
    python run.py \
         --tokenizer_dir=./Qwen-VL-Chat \
         --qwen_engine_dir=trt_engines/Qwen-VL-7B-int4 \
         --vit_engine_dir=./plan/
    ```
   
    - int4 gptq run
    ```bash
    python run.py \
        --tokenizer_dir=./Qwen-VL-Chat \
        --qwen_engine_dir=trt_engines/Qwen-VL-7B-int4-gptq \
        --vit_engine_dir=./plan/ 
    ```
