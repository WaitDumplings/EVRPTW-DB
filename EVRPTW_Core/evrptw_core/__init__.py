"""Shared schema, loading, validation, and metrics for EVRPTW-DB."""

from evrptw_core.io import load_instance, load_solution, save_solution
from evrptw_core.schema import EVRPTWInstance, EVRPTWSolution, ValidationResult, merge_route_sequences, solution_route_sequence
from evrptw_core.validation import validate_instance_structure

__all__ = [
    "EVRPTWInstance",
    "EVRPTWSolution",
    "merge_route_sequences",
    "solution_route_sequence",
    "ValidationResult",
    "load_instance",
    "load_solution",
    "save_solution",
    "validate_instance_structure",
]
