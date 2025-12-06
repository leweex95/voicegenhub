"""Placeholder hooks for future music generation support."""


class MusicGenerationError(NotImplementedError):
    """Indicates that music generation is not implemented yet."""


class MusicGenerator:
    """Lightweight stub to hold future music-generation logic."""

    def __init__(self, style: str = "cinematic") -> None:
        self.style = style

    def generate(self, description: str, duration: int) -> None:
        """Stubbed method until music generation is implemented."""
        raise MusicGenerationError(
            "Music generation support is coming soon."
        )
