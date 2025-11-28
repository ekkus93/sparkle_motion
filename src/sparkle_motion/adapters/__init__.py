"""Adapters package for model/tool integrations.

This package contains light-weight scaffolds for heavy model adapters. Each
adapter keeps heavy imports inside methods so unit tests and static analysis
don't require model packages to be installed.
"""
__all__ = ["WanAdapter", "DiffusersAdapter", "Wav2LipAdapter", "TTSAdapter"]
