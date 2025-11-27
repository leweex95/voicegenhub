"""
Microsoft Edge TTS Provider

High-quality TTS provider using Microsoft Edge's Text-to-Speech service.
Supports multiple voices, languages, and SSML markup.
"""

import asyncio
import os
import re
import sys
import tempfile
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
import edge_tts

from ..utils.logger import get_logger
from .base import (
    AudioFormat,
    ProviderCapabilities,
    TTSError,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    Voice,
    VoiceGender,
    VoiceNotFoundError,
    VoiceType,
)

logger = get_logger(__name__)


def _ensure_windows_event_loop():
    """
    Ensure Windows uses SelectorEventLoop to fix aiodns compatibility issues.

    On Windows, the default ProactorEventLoop doesn't work well with aiodns,
    which is used by edge-tts. This function switches to SelectorEventLoop.
    """
    if sys.platform == "win32":
        # Force SelectorEventLoop on Windows for aiodns compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger.debug("Set Windows SelectorEventLoop policy for aiodns compatibility")


def _patch_edge_tts_for_401_errors():
    """
    Monkey-patch edge-tts to handle 401 errors in addition to 403 errors.

    Microsoft's API now returns 401 Unauthorized instead of 403 Forbidden,
    but edge-tts only handles 403. This patch adds 401 handling to both
    list_voices() and Communicate.stream().
    """
    import ssl

    import certifi
    import edge_tts.communicate
    import edge_tts.voices
    from edge_tts.drm import DRM

    # Store reference to the private __list_voices function
    __list_voices = edge_tts.voices.__list_voices

    # Patch list_voices to handle 401 errors
    async def patched_list_voices(*, connector=None, proxy=None):
        """
        Patched version of list_voices that handles both 401 and 403 errors.
        """
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(
            connector=connector, trust_env=True
        ) as session:
            try:
                data = await __list_voices(session, ssl_ctx, proxy)
            except aiohttp.ClientResponseError as e:
                # Handle both 401 and 403 errors with clock skew correction
                if e.status not in (401, 403):
                    raise

                DRM.handle_client_response_error(e)
                data = await __list_voices(session, ssl_ctx, proxy)
        return data

    edge_tts.list_voices = patched_list_voices

    # Patch Communicate.stream to handle 401 errors
    async def patched_stream(self):
        """
        Patched version of Communicate.stream that handles both 401 and 403 errors.
        """
        if self.state.get("stream_was_called", False):
            raise RuntimeError("stream can only be called once.")
        self.state["stream_was_called"] = True

        # Stream the audio and metadata from the service.
        for self.state["partial_text"] in self.texts:
            try:
                async for message in self._Communicate__stream():
                    yield message
            except aiohttp.ClientResponseError as e:
                # Handle both 401 and 403 errors with clock skew correction
                if e.status not in (401, 403):
                    raise

                DRM.handle_client_response_error(e)
                async for message in self._Communicate__stream():
                    yield message

    edge_tts.communicate.Communicate.stream = patched_stream

    logger.info("Applied edge-tts 401 error handling patch")


# Apply the patch when the module is loaded
_patch_edge_tts_for_401_errors()


class EdgeTTSProvider(TTSProvider):
    """
    Microsoft Edge TTS provider implementation.

    Uses the edge-tts library to provide high-quality text-to-speech
    synthesis with Microsoft's neural voices.
    """

    def __init__(self, name: str = "edge", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._voices_cache: Optional[List[Voice]] = None
        self._raw_voices_cache: Optional[List[Dict]] = None
        self._rate_limit_delay = config.get("rate_limit_delay", 0.1) if config else 0.1
        self._initialization_failed = False
        self._max_retries = config.get("max_retries", 3) if config else 3
        self._retry_delay = config.get("retry_delay", 1.0) if config else 1.0

    @property
    def provider_id(self) -> str:
        return "edge"

    @property
    def display_name(self) -> str:
        return "Microsoft Edge TTS"

    async def initialize(self) -> None:
        """Initialize the Edge TTS provider with lazy loading."""
        # Ensure correct event loop on Windows for aiodns compatibility
        _ensure_windows_event_loop()

        # Don't fail initialization if the service is temporarily unavailable
        # Instead, mark it and try again during actual synthesis
        try:
            # Try to pre-fetch voices with retry logic
            await self._fetch_voices_with_retry()
            logger.info("Edge TTS provider initialized successfully")
            self._initialization_failed = False
        except Exception as e:
            logger.warning(
                f"Edge TTS initialization failed, will retry on first use: {e}"
            )
            # Don't raise error - allow lazy initialization
            self._initialization_failed = True

    def _parse_ssml_content(self, ssml_text: str) -> tuple[str, dict]:
        """
        Parse SSML content and extract clean text with prosody settings.

        Args:
            ssml_text: SSML markup text

        Returns:
            Tuple of (clean_text, prosody_settings)
        """
        try:
            # Simple approach: extract text content and basic prosody
            text = ssml_text
            prosody_settings = {}

            logger.info(f"SSML parsing input: {text}")

            # Extract rate from prosody tags (handle both quoted and unquoted)
            rate_match = re.search(r'<prosody[^>]*rate=(?:"([^"]*)"|(\w+))[^>]*>', text)
            if rate_match:
                rate_val = rate_match.group(1) or rate_match.group(2)
                logger.info(f"Found rate value: {rate_val}")
                if rate_val == "slow":
                    prosody_settings["rate"] = "-30%"
                elif rate_val == "fast":
                    prosody_settings["rate"] = "+30%"
                else:
                    prosody_settings["rate"] = rate_val

            # Extract pitch from prosody tags
            pitch_match = re.search(
                r'<prosody[^>]*pitch=(?:"([^"]*)"|(\w+))[^>]*>', text
            )
            if pitch_match:
                prosody_settings["pitch"] = pitch_match.group(1) or pitch_match.group(2)

            # Extract volume from prosody tags
            volume_match = re.search(
                r'<prosody[^>]*volume=(?:"([^"]*)"|(\w+))[^>]*>', text
            )
            if volume_match:
                prosody_settings["volume"] = volume_match.group(
                    1
                ) or volume_match.group(2)

            # Remove all XML/SSML tags to get clean text
            clean_text = re.sub(r"<[^>]+>", "", text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            logger.info(
                f"SSML parsing output: text='{clean_text}', prosody={prosody_settings}"
            )

            return clean_text, prosody_settings

        except Exception as e:
            logger.warning(f"SSML parsing failed: {e}, using regex fallback")
            # Remove XML tags as fallback
            clean_text = re.sub(r"<[^>]+>", "", ssml_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text, {}
            raise TTSError(
                f"Edge TTS initialization failed: {str(e)}",
                error_code="PROVIDER_INIT_FAILED",
                provider=self.provider_id,
            )

    async def _fetch_voices_with_retry(self) -> List[Dict]:
        """Fetch voices from Edge TTS API with retry logic.

        Clock skew correction for 401/403 errors is handled by the monkey patch
        applied to edge_tts.list_voices(). This method provides a retry loop with
        exponential backoff for any failures.
        """
        if self._raw_voices_cache is not None:
            logger.debug(f"Using cached voices ({len(self._raw_voices_cache)} voices)")
            return self._raw_voices_cache

        _ensure_windows_event_loop()

        last_error = None
        for attempt in range(self._max_retries):
            try:
                logger.info(
                    f"Fetching voices from Edge TTS API (attempt {attempt + 1}/{self._max_retries})"
                )
                voices = await edge_tts.list_voices()
                logger.info(f"Successfully fetched {len(voices)} voices from Edge TTS")
                self._raw_voices_cache = voices
                return voices
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to fetch voices (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        raise TTSError(
            f"Failed to fetch voices after {self._max_retries} attempts: {last_error}",
            error_code="VOICE_FETCH_FAILED",
            provider=self.provider_id,
        )

    async def _synthesize_with_retry(
        self,
        text: str,
        voice_name: str,
        rate_param: str,
        volume_param: str,
        pitch_param: str,
    ) -> bytes:
        """Synthesize audio with retry logic.

        Clock skew correction for 401/403 errors is handled by the monkey patch
        applied to edge_tts.Communicate.stream(). This method provides a retry loop with
        exponential backoff for any failures.
        """
        last_error = None
        for attempt in range(self._max_retries):
            try:
                logger.info(
                    f"Synthesizing audio (attempt {attempt + 1}/{self._max_retries})"
                )

                # Create TTS communication with native edge-tts parameters
                communicate = edge_tts.Communicate(
                    text,
                    voice_name,
                    rate=rate_param,
                    volume=volume_param,
                    pitch=pitch_param,
                )

                # Generate audio
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]

                if not audio_data:
                    raise TTSError(
                        "No audio data generated",
                        error_code="NO_AUDIO_GENERATED",
                        provider=self.provider_id,
                    )

                logger.info(
                    f"Successfully synthesized {len(audio_data)} bytes of audio"
                )
                return audio_data

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Synthesis failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    delay = self._retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        # All retries failed
        raise TTSError(
            f"Synthesis failed after {self._max_retries} attempts: {last_error}",
            error_code="SYNTHESIS_FAILED",
            provider=self.provider_id,
        )

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """
        Get available voices from Edge TTS.

        Args:
            language: Optional language filter

        Returns:
            List of available voices
        """
        try:
            if self._voices_cache:
                voices = self._voices_cache
            else:
                raw_voices = await self._fetch_voices_with_retry()

                voices = []
                for raw_voice in raw_voices:
                    voice = self._parse_edge_voice(raw_voice)
                    voices.append(voice)

                self._voices_cache = voices

            if language:
                filtered_voices = []
                for voice in voices:
                    if voice.language.startswith(
                        language.lower()
                    ) or voice.locale.startswith(language):
                        filtered_voices.append(voice)
                voices = filtered_voices

            logger.info(f"Retrieved {len(voices)} voices from Edge TTS")
            return voices

        except Exception as e:
            logger.error(f"Failed to get voices from Edge TTS: {e}")
            raise TTSError(
                f"Failed to retrieve voices: {str(e)}",
                error_code="VOICE_RETRIEVAL_FAILED",
                provider=self.provider_id,
            )

    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Edge TTS provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=True,
            supports_emotions=True,
            supports_styles=True,
            supports_speed_control=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_streaming=True,
            max_text_length=10000,  # Edge TTS has generous limits
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.MP3, AudioFormat.WAV],
            supported_sample_rates=[16000, 22050, 24000, 48000],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using Edge TTS.

        Args:
            request: TTS request parameters

        Returns:
            TTS response with audio data
        """
        import time

        # Ensure correct event loop on Windows
        _ensure_windows_event_loop()

        await self.validate_request(request)

        setup_start = time.perf_counter()

        try:
            # Get voice information
            voice_info = await self._get_voice_info(request.voice_id)
            if not voice_info:
                raise VoiceNotFoundError(
                    f"Voice {request.voice_id} not found",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            # Prepare text - handle SSML if present
            if request.ssml:
                # Parse SSML and extract clean text + prosody
                text, ssml_prosody = self._parse_ssml_content(request.text)
                logger.info(f"Parsed SSML: text='{text}', prosody={ssml_prosody}")

                # Use SSML prosody settings, but allow request parameters to override
                rate_param = ssml_prosody.get(
                    "rate", f"{int((request.speed - 1.0) * 100):+d}%"
                )
                volume_param = ssml_prosody.get(
                    "volume", f"{int((request.volume - 1.0) * 100):+d}%"
                )
                pitch_param = ssml_prosody.get(
                    "pitch", f"{int((request.pitch - 1.0) * 100):+d}Hz"
                )
            else:
                # Plain text
                text = request.text.strip()

                # Convert our speed/pitch/volume to edge-tts format
                rate_param = f"{int((request.speed - 1.0) * 100):+d}%"
                volume_param = f"{int((request.volume - 1.0) * 100):+d}%"
                pitch_param = f"{int((request.pitch - 1.0) * 100):+d}Hz"

            # Synthesize with retry logic
            inference_start = time.perf_counter()
            audio_data = await self._synthesize_with_retry(
                text, voice_info["Name"], rate_param, volume_param, pitch_param
            )
            inference_end = time.perf_counter()
            logger.debug(f"Edge TTS inference time: {inference_end - inference_start:.3f}s")

            # Convert format if needed
            if request.audio_format != AudioFormat.MP3:
                audio_data = await self._convert_audio_format(
                    audio_data, AudioFormat.MP3, request.audio_format
                )

            # Calculate duration (approximate)
            duration = self._estimate_duration(request.text, request.speed)

            setup_end = time.perf_counter()
            logger.info(f"Edge TTS setup time: {setup_end - setup_start:.3f}s")

            response = TTSResponse(
                audio_data=audio_data,
                format=request.audio_format,
                sample_rate=request.sample_rate,
                duration=duration,
                voice_used=voice_info["Name"],
                metadata={
                    "provider": self.provider_id,
                    "voice_locale": voice_info.get("Locale"),
                    "voice_gender": voice_info.get("Gender"),
                    "original_format": AudioFormat.MP3.value,
                },
            )

            # Rate limiting
            if self._rate_limit_delay > 0:
                await asyncio.sleep(self._rate_limit_delay)

            return response

        except Exception as e:
            if isinstance(e, TTSError):
                raise

            logger.error(f"Edge TTS synthesis failed: {e}")
            raise TTSError(
                f"Speech synthesis failed: {str(e)}",
                error_code="SYNTHESIS_FAILED",
                provider=self.provider_id,
            )

    async def synthesize_streaming(
        self, request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech with streaming response.

        Args:
            request: TTS request parameters

        Yields:
            Audio data chunks
        """
        await self.validate_request(request)

        try:
            voice_info = await self._get_voice_info(request.voice_id)
            if not voice_info:
                raise VoiceNotFoundError(
                    f"Voice {request.voice_id} not found",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            # Prepare text - handle SSML if present
            if request.ssml:
                # Parse SSML and extract clean text + prosody
                text, ssml_prosody = self._parse_ssml_content(request.text)
                logger.info(f"Parsed SSML: text='{text}', prosody={ssml_prosody}")

                # Use SSML prosody settings, but allow request parameters to override
                rate_param = ssml_prosody.get(
                    "rate", f"{int((request.speed - 1.0) * 100):+d}%"
                )
                volume_param = ssml_prosody.get(
                    "volume", f"{int((request.volume - 1.0) * 100):+d}%"
                )
                pitch_param = ssml_prosody.get(
                    "pitch", f"{int((request.pitch - 1.0) * 100):+d}Hz"
                )
            else:
                # Plain text
                text = request.text.strip()

                # Convert our speed/pitch/volume to edge-tts format
                rate_param = f"{int((request.speed - 1.0) * 100):+d}%"
                volume_param = f"{int((request.volume - 1.0) * 100):+d}%"
                pitch_param = f"{int((request.pitch - 1.0) * 100):+d}Hz"

            # For streaming, we use a simpler approach with retry on failure
            last_error = None
            for attempt in range(self._max_retries):
                try:
                    # Create TTS communication with native edge-tts parameters
                    communicate = edge_tts.Communicate(
                        text,
                        voice_info["Name"],
                        rate=rate_param,
                        volume=volume_param,
                        pitch=pitch_param,
                    )

                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            yield chunk["data"]

                    # Rate limiting
                    if self._rate_limit_delay > 0:
                        await asyncio.sleep(self._rate_limit_delay)

                    # If we got here, streaming succeeded
                    return

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Streaming synthesis failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                    )
                    if attempt < self._max_retries - 1:
                        # Exponential backoff
                        delay = self._retry_delay * (2**attempt)
                        logger.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)

            # All retries failed
            raise TTSError(
                f"Streaming synthesis failed after {self._max_retries} attempts: {last_error}",
                error_code="STREAMING_FAILED",
                provider=self.provider_id,
            )

        except Exception as e:
            if isinstance(e, TTSError):
                raise

            logger.error(f"Edge TTS streaming failed: {e}")
            raise TTSError(
                f"Streaming synthesis failed: {str(e)}",
                error_code="STREAMING_FAILED",
                provider=self.provider_id,
            )

    async def health_check(self) -> bool:
        """Check if Edge TTS service is available."""
        try:
            # Try to get voices as health check with retry
            voices = await self._fetch_voices_with_retry()
            return len(voices) > 0

        except Exception as e:
            logger.error(f"Edge TTS health check failed: {e}")
            return False

    def _parse_edge_voice(self, raw_voice: Dict) -> Voice:
        """Parse Edge TTS voice data into our Voice format."""
        # Extract voice metadata
        name = raw_voice.get("Name", "")
        short_name = raw_voice.get("ShortName", name)
        locale = raw_voice.get("Locale", "")
        language = locale.split("-")[0] if locale else ""
        gender_str = raw_voice.get("Gender", "").lower()

        # Map gender
        gender = VoiceGender.NEUTRAL
        if gender_str == "male":
            gender = VoiceGender.MALE
        elif gender_str == "female":
            gender = VoiceGender.FEMALE

        # Determine voice type (Edge voices are neural)
        voice_type = VoiceType.NEURAL

        # Extract styles and emotions if available
        voice_tag = raw_voice.get("VoiceTag", {})
        content_categories = voice_tag.get("ContentCategories", [])
        voice_personalities = voice_tag.get("VoicePersonalities", [])

        # Estimate quality score based on available features
        quality_score = 0.8  # Base score for Edge TTS
        if content_categories:
            quality_score += 0.1
        if voice_personalities:
            quality_score += 0.1

        return Voice(
            id=short_name,
            name=short_name,
            language=language,
            locale=locale,
            gender=gender,
            voice_type=voice_type,
            provider=self.provider_id,
            sample_rate=24000,  # Edge TTS default
            description=f"Microsoft Edge TTS - {name}",
            emotions=None,  # Edge TTS doesn't explicitly list emotions
            styles=voice_personalities if voice_personalities else None,
            age_group=self._determine_age_group(voice_personalities),
            quality_score=min(quality_score, 1.0),
        )

    def _determine_age_group(self, personalities: List[str]) -> Optional[str]:
        """Determine age group from voice personalities."""
        if not personalities:
            return None

        personalities_str = " ".join(personalities).lower()
        if any(word in personalities_str for word in ["child", "young", "kid"]):
            return "child"
        elif any(word in personalities_str for word in ["adult", "mature"]):
            return "adult"
        elif any(word in personalities_str for word in ["elderly", "senior", "old"]):
            return "elderly"

        return "adult"  # Default

    async def _get_voice_info(self, voice_id: str) -> Optional[Dict]:
        """Get detailed voice information by ID using cached data."""
        try:
            voices = await self._fetch_voices_with_retry()
            for voice in voices:
                if voice.get("ShortName") == voice_id or voice.get("Name") == voice_id:
                    return voice
            return None
        except Exception as e:
            logger.error(f"Failed to get voice info for {voice_id}: {e}")
            return None

    async def _convert_audio_format(
        self, audio_data: bytes, from_format: AudioFormat, to_format: AudioFormat
    ) -> bytes:
        """Convert audio between formats using pydub."""
        if from_format == to_format:
            return audio_data

        try:
            from pydub import AudioSegment

            # Ensure formats are strings
            from_fmt = (
                from_format.value if hasattr(from_format, "value") else str(from_format)
            )
            to_fmt = to_format.value if hasattr(to_format, "value") else str(to_format)

            # Create temporary files with proper cleanup
            temp_in_path = None
            temp_out_path = None

            try:
                # Create input temp file
                with tempfile.NamedTemporaryFile(
                    suffix=f".{from_fmt}", delete=False
                ) as temp_in:
                    temp_in_path = temp_in.name
                    temp_in.write(audio_data)

                # Load with pydub
                audio = AudioSegment.from_file(temp_in_path, format=from_fmt)

                # Create output temp file
                with tempfile.NamedTemporaryFile(
                    suffix=f".{to_fmt}", delete=False
                ) as temp_out:
                    temp_out_path = temp_out.name

                # Export to target format
                audio.export(temp_out_path, format=to_fmt)

                # Read the converted audio
                with open(temp_out_path, "rb") as f:
                    converted_data = f.read()

                return converted_data

            finally:
                # Clean up temp files
                if temp_in_path and os.path.exists(temp_in_path):
                    try:
                        os.unlink(temp_in_path)
                    except OSError:
                        logger.warning(f"Could not delete temp file: {temp_in_path}")

                if temp_out_path and os.path.exists(temp_out_path):
                    try:
                        os.unlink(temp_out_path)
                    except OSError:
                        logger.warning(f"Could not delete temp file: {temp_out_path}")

        except ImportError:
            logger.warning("pydub not available, cannot convert audio format")
            return audio_data
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return audio_data

    def _estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate audio duration based on text length and speed."""
        # Rough estimation: ~150 words per minute for normal speech
        words = len(text.split())
        base_duration = (words / 150.0) * 60.0  # seconds

        # Adjust for speed
        adjusted_duration = base_duration / speed

        return max(adjusted_duration, 0.1)  # Minimum 0.1 seconds
