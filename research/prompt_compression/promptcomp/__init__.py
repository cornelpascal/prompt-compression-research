"""Portable, whole-prompt compression. Model imports are lazy."""
from .compressors import compress
from .schema import CompressionRequest, CompressionResult

__all__ = ["compress", "CompressionRequest", "CompressionResult"]
__version__ = "0.1.0"

