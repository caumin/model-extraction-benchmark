"""Upstream-faithful components for Black-Box Ripper.

This subpackage vendors (with minimal modifications) the generator architectures
used by the official implementation of:
  https://github.com/antoniobarbalau/black-box-ripper

We keep module/parameter names stable so that the official checkpoints can be
loaded via `state_dict` without key mismatches.
"""

from mebench.models.blackbox_ripper.factory import (
    create_blackbox_ripper_generator,
    load_blackbox_ripper_generator_weights,
)

__all__ = [
    "create_blackbox_ripper_generator",
    "load_blackbox_ripper_generator_weights",
]
