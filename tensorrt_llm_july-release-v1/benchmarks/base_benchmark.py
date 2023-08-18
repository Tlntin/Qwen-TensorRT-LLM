import json
import os

import torch

import tensorrt_llm


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    with open(path, 'wb') as f:
        f.write(bytearray(engine))


class BaseBenchmark(object):

    def __init__(self, engine_dir, model_name, dtype, output_dir):
        self.engine_dir = engine_dir
        self.model_name = model_name
        self.dtype = dtype
        self.output_dir = output_dir
        self.runtime_rank = tensorrt_llm.mpi_rank()
        self.world_size = tensorrt_llm.mpi_world_size()
        if engine_dir is not None:
            # Read config from engine directory
            config_path = os.path.join(engine_dir, 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            self.dtype = self.config['builder_config']['precision']
            world_size = self.config['builder_config']['tensor_parallel']
            assert world_size == self.world_size, \
                (f'Engine world size ({world_size}) != Runtime world size ({self.world_size})')

        self.engine_name = get_engine_name(model_name, dtype, self.world_size,
                                           self.runtime_rank)
        self.runtime_mapping = tensorrt_llm.Mapping(self.world_size,
                                                    self.runtime_rank)
        torch.cuda.set_device(self.runtime_rank %
                              self.runtime_mapping.gpus_per_node)

    def get_config(self):
        raise NotImplementedError

    def prepare_inputs(self, config):
        raise NotImplementedError

    def run(self, inputs, config):
        raise NotImplementedError

    def report(self, config, latency):
        raise NotImplementedError
