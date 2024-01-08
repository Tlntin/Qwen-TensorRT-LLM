# Guide to QWen-VL pipeline
1. Download Qwen-VL
    ```bash
    git lfs install
    git clone https://huggingface.co/Qwen/Qwen-VL
    ```
2. ViT
- Generate ONNX model and TRT engine for ViT
    ```bash
    python vit_onnx_trt.py --pretrained_model_path ./qwen/Qwen-VL
    ```
    The exported ONNX files lies in `./onnx/visual_encoder` and the built engine lie in `./plan/visual_encoder`. And you have onnx files already and convert TRT engine only, use:
    ```bash
    python vit_onnx_trt.py --pretrained_model_path ./qwen/Qwen-VL --only_trt
    ```
    Moreover, it will save test image tensor to `image.pt` and visual query tokens to `query_tokens.pt` for later pipeline inference.

2. QwenVL

- Build TRT-LLM engines (only need to add --max_prompt_embedding_table_size)

    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3 build.py  \
	--hf_model_dir=./qwen/Qwen-VL/ \
	--dtype float16 --max_batch_size 8 \
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

3. Qwen-VL-xxx-Int4
    **Note**: load weight from orgin，that is a example
    ```bash
    python3 build.py \
	--hf_model_dir=./Qwen-VL-Chat-Int4/ \
	--dtype float16 --max_batch_size 1 \
	--max_input_len 6144 --max_new_tokens 2048 \
	--remove_input_padding \
	--use_gpt_attention_plugin float16 \
	--use_gemm_plugin float16 --enable_context_fmha \
	--use_rmsnorm_plugin --log_level error \
	--use_lookup_plugin float16 \
	--max_prompt_embedding_table_size 6144 \
	--use_weight_only \
    --weight_only_precision int4_gptq \
    --per_group \
    --quant_ckpt_path ./Qwen-VL-Chat-Int4/ \
	--output_dir=trt_engines/Qwen-VL-7B-fp16 
    ```

4. Run Qwen-VL pipeline
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run.py \
	--tokenizer_dir=./qwen/Qwen-VL \
	--qwen_engine_dir=./trt_engines/Qwen-VL-7B-fp16/ \
	--vit_engine_dir=./plan/
    ```
