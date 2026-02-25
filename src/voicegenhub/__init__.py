"""
VoiceGenHub - Universal Text-to-Speech Library

A comprehensive TTS library supporting multiple providers with advanced features:
- Multi-provider support (Edge TTS, Google Cloud TTS, Azure, etc.)
- Advanced voice management and selection
- Audio processing and effects
- Intelligent caching system
- REST API for service integration
- Robust error handling and resilience

Example usage:
    from voicegenhub import VoiceGenHub

    tts = VoiceGenHub()
    audio = await tts.generate("Hello, world!", voice="en-US-AriaNeural")
"""

__version__ = "1.1.4"
__author__ = "leweex95"
__email__ = "csibi.levente14@gmail.com"

import os
# Set attention implementation to eager before any imports to prevent SDPA warnings
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'eager'

# Apply CPU compatibility patches early to prevent import-time crashes in dependencies
try:
    from .utils.compatibility import apply_cpu_compatibility_patches
    apply_cpu_compatibility_patches()
except Exception:
    # Fail silently to avoid breaking the whole library if the tool itself has issues
    pass

# Import core classes when available
try:
    from .config.settings import Settings  # noqa: F401
    from .core.engine import VoiceGenHub  # noqa: F401
    from .core.voice import VoiceSelector  # noqa: F401
    from .providers.base import TTSProvider, Voice  # noqa: F401

    __all__ = [
        "VoiceGenHub",
        "VoiceSelector",
        "Voice",
        "TTSProvider",
        "Settings",
    ]
except ImportError as e:
    print("VoiceGenHub import error:", e)
    __all__ = []
