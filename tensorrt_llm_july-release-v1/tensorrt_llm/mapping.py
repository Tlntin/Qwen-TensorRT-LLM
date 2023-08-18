from dataclasses import dataclass, field


@dataclass
class Mapping:
    world_size: int = 1
    rank: int = 0
    gpus_per_node: int = 8
    tp_size: int = field(init=False)
    tp_group: list = field(init=False)

    def __post_init__(self):
        self.tp_size = self.world_size
        self.tp_group = list(range(self.tp_size))
