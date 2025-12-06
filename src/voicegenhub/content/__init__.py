"""Structured entry points for VoiceGenHub content generators."""

from __future__ import annotations

from enum import Enum

from .effect import (
    EffectConfigurationError,
    EffectGenerationError,
    EffectGenerationResult,
    StableAudioEffectGenerator,
)
from .music import MusicGenerator, MusicGenerationError


class ContentType(Enum):
    """Supported content categories."""

    TTS = "tts"
    MUSIC = "music"
    EFFECT = "effect"


__all__ = [
    "ContentType",
    "StableAudioEffectGenerator",
    "EffectGenerationResult",
    "EffectGenerationError",
    "EffectConfigurationError",
    "MusicGenerator",
    "MusicGenerationError",
]
