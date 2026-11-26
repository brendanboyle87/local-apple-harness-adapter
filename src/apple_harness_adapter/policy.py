"""Compatibility imports for existing Inkling profiling scripts."""

from .backends.inkling import InklingPolicy, checkpoint_decoder_layers

__all__ = ["InklingPolicy", "checkpoint_decoder_layers"]
