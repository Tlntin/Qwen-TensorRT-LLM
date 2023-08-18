import collections
import contextlib

import numpy as np
import tensorrt as trt

from ._common import set_network
from .logger import logger
from .plugin import PluginConfig


class _UniqueNameGenerator(object):

    def __init__(self, prefix=''):
        self.ids = collections.defaultdict(int)
        self.prefix = prefix

    def __call__(self, key, module_name=''):
        if module_name != '':
            module_name = module_name.replace(".", "/")
            key = module_name + '/' + key
        tmp = self.ids[key]
        self.ids[key] += 1
        return f"{self.prefix}{key}_{tmp}"


class Network(object):

    def __init__(self, **kwargs):
        # intentionally use **kwargs, user should never call this ctor directly
        # use Builder.create_network() instead
        pass

    def _init(self, trt_network):
        self._trt_network = trt_network
        self._inputs = {}
        self._named_parameters = None
        # layer precision of a given scope, this is used together with precision(dtype) context manager
        self._dtype = None
        self._name_generator = _UniqueNameGenerator()
        self._plugin_config = PluginConfig()
        self._module_call_stack = _TrtLlmModuleCallStack()
        self._registered_ndarrays = []

        return self

    @property
    def dtype(self) -> trt.DataType:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: trt.DataType):
        assert isinstance(dtype, trt.DataType) or dtype is None
        self._dtype = dtype

    @property
    def trt_network(self) -> trt.INetworkDefinition:
        return self._trt_network

    @property
    def plugin_config(self) -> PluginConfig:
        return self._plugin_config

    def _add_input(self, tensor, name, dtype, shape, dim_range=None):
        assert isinstance(dtype, trt.DataType)
        tensor.trt_tensor = self.trt_network.add_input(
            name=name,
            shape=shape,
            dtype=dtype,
        )
        logger.debug(f'Add input: {name}, shape: {shape}, dtype: {dtype}')
        if dim_range is not None:
            for i, dim_name in enumerate(dim_range.keys()):
                tensor.trt_tensor.set_dimension_name(i, str(dim_name))
        self._inputs[name] = tensor

    def _mark_output(self, tensor, name, dtype):
        self.trt_network.mark_output(tensor.trt_tensor)
        tensor.trt_tensor.name = name
        tensor.trt_tensor.dtype = dtype
        logger.debug(f'Mark output: {name}, dtype: {dtype}')

    def set_named_parameters(self, named_parameters):
        self._named_parameters = named_parameters

    @property
    def named_parameters(self):
        return self._named_parameters

    def _set_layer_name(self, layer):
        layer_name = str(layer.type).split('.')[-1]
        current_module = self._module_call_stack.get_current_module()

        if layer.type == trt.LayerType.PLUGIN_V2:
            layer_name = '_'.join(
                [layer_name,
                 str(layer.plugin.plugin_type).split('.')[-1]])
        elif layer.type in [
                trt.LayerType.UNARY, trt.LayerType.REDUCE,
                trt.LayerType.ELEMENTWISE
        ]:
            layer_name = '_'.join([layer_name, str(layer.op).split('.')[-1]])

        layer.name = self._name_generator(layer_name, current_module)
        for idx in range(layer.num_outputs):
            # TRT initializes tensor names from the initial layer's name when the layer is created,
            # and does not update tensor names when layer name changed by application, needs to
            # change the tensor name to align with the new layer name for better debugging
            layer.get_output(idx).name = f"{layer.name}_output_{idx}"

    def register_ndarray(self, ndarray: np.ndarray) -> None:
        self._registered_ndarrays.append(ndarray)


@contextlib.contextmanager
def net_guard(network):
    assert isinstance(
        network, Network
    ), f"Invalid network, can only guard Network instance, got: {network}"
    set_network(network)
    yield
    set_network(None)


class _TrtLlmModuleCallStack(object):
    call_stack = []
    module_name_map = {}

    def __init__(self):
        super().__init__()
        self.mod_names_set = False

    def module_names_set(self):
        return self.mod_names_set

    def set_module_names(self, top_level_module):
        assert top_level_module, "Expected a top level module"
        for name, mod in top_level_module.named_modules(
                prefix=top_level_module._get_name()):
            if mod not in self.module_name_map:
                self.module_name_map[mod] = name
        self.mod_names_set = True
        return

    def get_current_module(self):
        mod_name = ''
        if len(self.call_stack):
            mod_name = self.call_stack[-1]
        return mod_name

    def get_mod_name(self, mod_obj):
        name = ''
        if mod_obj in self.module_name_map:
            name = self.module_name_map[mod_obj]
        return name

    def get_stack(self):
        return self.call_stack

    @contextlib.contextmanager
    def call_stack_mgr(self):
        call_stack = self.get_stack()
        try:
            yield call_stack
        finally:
            call_stack.pop()
