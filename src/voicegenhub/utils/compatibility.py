"""Compatibility utilities for problematic dependencies."""

import sys
import importlib.metadata
import os
from .logger import get_logger

logger = get_logger(__name__)

def apply_cpu_compatibility_patches():
    """Apply patches to ensure stability on CPU-only environments."""

    # 1. Mock torchcodec if missing or corrupt. This is a common failure point for Transformers >= 4.51 on CPU.
    # Handling PackageNotFoundError for missing, or Exception (e.g. IndexError) for corrupted installs.
    is_broken_or_missing = False
    try:
        importlib.metadata.version("torchcodec")
    except (importlib.metadata.PackageNotFoundError, Exception):
        is_broken_or_missing = True

    if is_broken_or_missing or "torchcodec" not in sys.modules:
        logger.info("torchcodec not found or corrupt, applying compatibility mocks for Transformers/AudioUtils")

        # Mocking sys.modules
        import types
        from importlib.machinery import ModuleSpec

        mock_codec = types.ModuleType("torchcodec")
        mock_codec.__version__ = "0.9.1"
        mock_codec.__spec__ = ModuleSpec("torchcodec", None)

        class Frame: pass
        class Decoder:
            def __init__(self, *args, **kwargs): pass

        mock_codec.Frame = Frame
        mock_codec.Decoder = Decoder

        sys.modules["torchcodec"] = mock_codec

        # Mocking importlib.metadata.version
        # Save a reference to the original version function if not already patched
        if not hasattr(importlib.metadata, "_original_version"):
            importlib.metadata._original_version = importlib.metadata.version

            def patched_version(package_name):
                if package_name == "torchcodec":
                    return "0.9.1"
                return importlib.metadata._original_version(package_name)

            importlib.metadata.version = patched_version
            # Some versions of Python/importlib might need patching in different places
            # but this is the most common one.

    # 2. Performance/Stability environment variables
    # Only set if not already present
    if "TRANSFORMERS_ATTENTION_IMPLEMENTATION" not in os.environ:
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

    # 3. Patching torch specifically for CPU stability
    # We do this lazily to avoid triggering heavy torch imports if they haven't happened yet
    if "torch" in sys.modules:
        _patch_torch_cuda(sys.modules["torch"])

def _patch_torch_cuda(torch):
    """Specific patches for torch when it's already loaded."""
    if not torch.cuda.is_available():
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Patch is_bf16_supported if it exists in torch.cuda
        if hasattr(torch.cuda, "is_bf16_supported"):
            # Some libs expect this to return False on CPU instead of crashing
            pass
        else:
            try:
                torch.cuda.is_bf16_supported = lambda: False
            except Exception:
                pass

def ensure_torchcodec():
    """Specific check for torchcodec to satisfy Transformers >= 4.51."""
    try:
        importlib.metadata.version("torchcodec")
    except importlib.metadata.PackageNotFoundError:
        # If we got here, apply patches if not already done
        apply_cpu_compatibility_patches()
