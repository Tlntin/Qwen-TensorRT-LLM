import os
import argparse
from run import get_model
from run import Qwen2ForCausalLMGenerationSession
from default_config import default_config

now_dir = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=default_config.max_new_tokens)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=default_config.engine_dir,
    )
    parser.add_argument(
        '--tokenizer_dir',
        type=str,
        default=default_config.tokenizer_dir,
        help="Directory containing the tokenizer.model."
    )
    parser.add_argument(
        '--stream',
        type=bool,
        default=True,
        help="return text with stream")
    return parser.parse_args()


if __name__ == "__main__":
    # get model info
    args = parse_arguments()
    (
        model_config, sampling_config, runtime_mapping, runtime_rank,
        serialize_path, remove_input_padding, 
        tokenizer, eos_token_id, pad_token_id, stop_token_ids
    ) = get_model(args.tokenizer_dir, args.engine_dir, args.log_level)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = Qwen2ForCausalLMGenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping,
    )
    history = []
    response = ''
    print("欢迎使用Qwen聊天机器人，输入exit退出，输入clear清空历史记录")
    while True:
        input_text = input("Input: ")
        if input_text in ["exit", "quit", "exit()", "quit()"]:
            break
        if input_text == 'clear':
            history = []
            continue
        if not args.stream:
            response = decoder.chat(
                pad_token_id=pad_token_id,
                tokenizer=tokenizer,
                sampling_config=sampling_config,
                input_text=input_text, 
                history=history,
                max_new_tokens=args.max_new_tokens,
            )
            print(f'Output: {response[0]}')
        else:
            print("Output: ", end='')

            response = ""
            for new_text in decoder.chat_stream(
                stop_token_ids=stop_token_ids,
                pad_token_id=pad_token_id,
                tokenizer=tokenizer,
                sampling_config=sampling_config,
                input_text=input_text,
                history=history,
                max_new_tokens=args.max_new_tokens,
            ):
                print(new_text[0], end='', flush=True)
                response += new_text[0]
            print("")
        history.append((input_text, response))