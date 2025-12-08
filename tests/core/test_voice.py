"""Unit tests for VoiceSelector."""
from unittest.mock import AsyncMock

import pytest

from voicegenhub.core.voice import VoiceSelector
from voicegenhub.providers.base import Voice, VoiceGender, VoiceType


class TestVoiceSelector:
    """Unit tests for VoiceSelector."""

    @pytest.fixture
    def sample_voices(self):
        """Create sample voices for testing."""
        return [
            Voice(
                id="en-US-AriaNeural",
                name="Aria",
                language="en",
                locale="en-US",
                gender=VoiceGender.FEMALE,
                voice_type=VoiceType.NEURAL,
                provider="edge",
            ),
            Voice(
                id="en-GB-SoniaNeural",
                name="Sonia",
                language="en",
                locale="en-GB",
                gender=VoiceGender.FEMALE,
                voice_type=VoiceType.NEURAL,
                provider="edge",
            ),
            Voice(
                id="es-ES-ElviraNeural",
                name="Elvira",
                language="es",
                locale="es-ES",
                gender=VoiceGender.FEMALE,
                voice_type=VoiceType.NEURAL,
                provider="edge",
            ),
        ]

    @pytest.fixture
    def mock_providers(self, sample_voices):
        """Mock providers that return voices."""
        provider1 = AsyncMock()
        provider1.get_voices.return_value = sample_voices[:2]  # en voices
        provider2 = AsyncMock()
        provider2.get_voices.return_value = sample_voices[2:]  # es voice
        return [provider1, provider2]

    def test_init_empty(self):
        """Test VoiceSelector initialization with no providers."""
        selector = VoiceSelector()
        assert selector.providers == []

    def test_init_with_providers(self, mock_providers):
        """Test VoiceSelector initialization with providers."""
        selector = VoiceSelector(mock_providers)
        assert selector.providers == mock_providers

    @pytest.mark.asyncio
    async def test_get_all_voices(self, mock_providers, sample_voices):
        """Test getting all voices from providers."""
        selector = VoiceSelector(mock_providers)
        voices = await selector.get_all_voices()

        assert len(voices) == 3
        assert voices == sample_voices

        # Verify both providers were called
        mock_providers[0].get_voices.assert_called_once()
        mock_providers[1].get_voices.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_voices_provider_error(self, mock_providers):
        """Test handling provider errors when getting voices."""
        mock_providers[0].get_voices.side_effect = Exception("Provider error")

        selector = VoiceSelector(mock_providers)

        with pytest.raises(Exception, match="Provider error"):
            await selector.get_all_voices()

    @pytest.mark.asyncio
    async def test_select_voice_no_language(self, mock_providers, sample_voices):
        """Test voice selection without language preference."""
        selector = VoiceSelector(mock_providers)
        voice = await selector.select_voice()

        assert voice == sample_voices[0]  # First voice

    @pytest.mark.asyncio
    async def test_select_voice_with_language_match(self, mock_providers, sample_voices):
        """Test voice selection with language that matches."""
        selector = VoiceSelector(mock_providers)
        voice = await selector.select_voice(language="en")

        assert voice == sample_voices[0]  # First English voice

    @pytest.mark.asyncio
    async def test_select_voice_with_language_no_match(self, mock_providers, sample_voices):
        """Test voice selection with language that doesn't match."""
        selector = VoiceSelector(mock_providers)
        voice = await selector.select_voice(language="fr")  # No French voices

        assert voice == sample_voices[0]  # Fallback to first voice

    @pytest.mark.asyncio
    async def test_select_voice_locale_match(self, mock_providers, sample_voices):
        """Test voice selection with locale matching."""
        selector = VoiceSelector(mock_providers)
        voice = await selector.select_voice(language="en-GB")

        # Currently only checks language, not locale, so returns first en voice
        assert voice == sample_voices[0]  # US voice

    @pytest.mark.asyncio
    async def test_select_voice_no_voices(self):
        """Test voice selection when no voices available."""
        mock_provider = AsyncMock()
        mock_provider.get_voices.return_value = []

        selector = VoiceSelector([mock_provider])
        voice = await selector.select_voice()

        assert voice is None
