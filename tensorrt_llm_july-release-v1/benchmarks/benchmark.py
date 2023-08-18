import argparse
from multiprocessing import Process, Queue

import torch
from bert_benchmark import BERTBenchmark
from gpt_benchmark import GPTBenchmark
from mem_monitor import mem_monitor

from tensorrt_llm.logger import logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Benchmark TensorRT-LLM models.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="gpt_350m",
                        choices=[
                            'gpt_350m', 'gpt_175b', 'gpt_350m_sq_per_tensor',
                            'gpt_350m_sq_per_token_channel', 'opt_350m',
                            'opt_66b', 'llama_7b', 'llama_30b', 'gptj_6b',
                            'gptneox_20b', 'bert_base', 'bert_large',
                            'chatglm_6b'
                        ],
                        help='Specify model you want to benchmark.')
    parser.add_argument(
        '--mode',
        type=str,
        default="plugin",
        choices=['ootb', 'plugin'],
        help=
        ('Choose mode between ootb/plugin. '
         '\"ootb\" means the engines will be built without any plugins, '
         'while \"plugin\" means the engines will be built with tuned recipe of using plugins.'
         ))

    parser.add_argument('--batch_size',
                        type=str,
                        default="8",
                        help=('Specify batch size(s) you want to benchmark. '
                              'Multiple batch sizes can be separated by \";\", '
                              'example: \"1;8;64\".'))
    parser.add_argument(
        '--input_len',
        type=str,
        default="128",
        help=('Specify input length(s) you want to benchmark, '
              'this option is mainly for BERT. '
              'Multiple input lengths can be separated by \";\", '
              'example: \"20;60;128\".'))
    parser.add_argument(
        '--input_output_len',
        type=str,
        default="128,20",
        help=('Specify input-output length(s) you want to benchmark, '
              'this option is mainly for GPT and GPT-like models. '
              'Multiple input lengths can be separated by \";\", '
              'example: \"60,20;128,20\".'))
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'],
                        help='Choose data type between float16/float32.')
    parser.add_argument(
        '--refit',
        default=False,
        action="store_true",
        help=
        'If this option is specified, a refit flag is added to TensorRT engines.'
    )

    parser.add_argument('--num_beams',
                        type=int,
                        default="1",
                        help=('Specify number of beams you want to benchmark.'))
    parser.add_argument('--top_k',
                        type=int,
                        default="1",
                        help=('Specify Top-K value of decoding.'))
    parser.add_argument('--top_p',
                        type=float,
                        default="0",
                        help=('Specify Top-P value of decoding.'))

    parser.add_argument(
        '--log_level',
        type=str,
        default="error",
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'],
        help=
        'Choose log level between verbose/info/warning/error/internal_error.')
    parser.add_argument(
        '--warm_up',
        type=int,
        default=2,
        help='Specify warm up iterations before benchmark starts.')
    parser.add_argument(
        '--num_runs',
        type=int,
        default=10,
        help='Specify number of iterations to run during benchmarking.')

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help=
        'If this option is specified, TensorRT engines will be saved to engine_dir.'
    )
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=None,
        help=
        ('If this option is specified, instead of building engines on-air before benchmarking, '
         'the engines contained in the engine_dir will be used.'))
    return parser.parse_args()


def main(args):
    logger.set_level(args.log_level)

    # Batch size
    batch_size_options = args.batch_size.split(';')
    batch_size_options = [int(i) for i in batch_size_options]
    # Input length (for BERT-like models)
    input_len_options = args.input_len.split(';')
    input_len_options = [int(i) for i in input_len_options]
    # Input-output length combination (for GPT-like models)
    in_out_len_options = args.input_output_len.split(';')
    in_out_len_options = [[int(i) for i in io.split(',')]
                          for io in in_out_len_options]

    if args.model in [
            'gpt_350m',
            'gpt_175b',
            'gpt_350m_sq_per_tensor',
            'gpt_350m_sq_per_token_channel',
            'opt_350m',
            'opt_66b',
            'llama_7b',
            'llama_30b',
            'gptj_6b',
            'gptneox_20b',
            'chatglm_6b',
    ]:
        benchmarker = GPTBenchmark(args.engine_dir, args.model, args.mode,
                                   batch_size_options, in_out_len_options,
                                   args.dtype, args.refit, args.num_beams,
                                   args.top_k, args.top_p, args.output_dir)
    elif args.model in ['bert_base', 'bert_large']:
        benchmarker = BERTBenchmark(args.engine_dir, args.model, args.mode,
                                    batch_size_options, input_len_options,
                                    args.dtype, args.output_dir)
    else:
        raise Exception(f'Unexpected model: {args.model}')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for config in benchmarker.get_config():
        inputs = benchmarker.prepare_inputs(config)
        torch.cuda.empty_cache()
        latencies = []

        # Launch a subprocess to monitor memory usage
        q1 = Queue()  # q1 is used for sending signal to subprocess
        q2 = Queue()  # q2 is used for receiving results from subprocess
        p = Process(target=mem_monitor, args=(q1, q2))
        p.start()

        try:
            # Warm up
            for _ in range(args.warm_up):
                benchmarker.run(inputs, config)
            logger.info('Warm up done. Start benchmarking.')

            for _ in range(args.num_runs):
                start.record()
                benchmarker.run(inputs, config)
                end.record()

                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))

        except Exception as e:
            p.kill()
            raise e

        q1.put(1)
        peak_gpu_used = q2.get()
        p.join()

        latency = round(sum(latencies) / args.num_runs, 3)
        latencies.sort()
        percentile95 = round(latencies[int(args.num_runs * 0.95)], 3)
        percentile99 = round(latencies[int(args.num_runs * 0.99)], 3)
        benchmarker.report(config, latency, percentile95, percentile99,
                           peak_gpu_used)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
