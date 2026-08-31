"""Optional online-environment adapters."""

try:
    from .minerl1 import MineRL1EnvWrapper, MineRLModelTick, observation_to_tensor
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ = ["MineRL1EnvWrapper", "MineRLModelTick", "observation_to_tensor"]

