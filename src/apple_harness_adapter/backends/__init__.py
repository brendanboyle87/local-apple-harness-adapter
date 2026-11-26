"""Model-family selection and pinned defaults."""

DEFAULT_MODELS = {
    "inkling": "mlx-community/Inkling-2-Mini-4bit",
    "qwen3": "Qwen/Qwen3-1.7B-MLX-bf16",
    "lfm3": "LiquidAI/LFM3-1.7B-MLX-bf16",
}

DEFAULT_REVISIONS = {
    DEFAULT_MODELS["inkling"]: "9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62",
    DEFAULT_MODELS["qwen3"]: "720c04346ea2b095c801ebbd545c109230964cd4",
    DEFAULT_MODELS["lfm3"]: "52041852d5b5c59bfe120c075d2744ab97959d91",
    "Qwen/Qwen3-0.6B-MLX-bf16": "bc82a1060abf25e90be9782b12c00fa55d9bf542",
}


def backend_class(name: str):
    if name == "inkling":
        from .inkling import InklingPolicy

        return InklingPolicy
    if name == "qwen3":
        from .qwen3 import Qwen3Policy

        return Qwen3Policy
    if name == "lfm3":
        from .lfm3 import LFM3Policy

        return LFM3Policy
    raise ValueError(f"Unknown model backend {name!r}")
