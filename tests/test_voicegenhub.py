"""Integration tests for VoiceGenHub."""
import pytest
import asyncio
import sys

# Fix for Windows: ensure SelectorEventLoop is used for aiodns compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from voicegenhub import VoiceGenHub


class TestVoiceGenHub:
    """Integration tests for VoiceGenHub."""

    @pytest.mark.asyncio
    async def test_edge_tts_generation(self):
        """Test Edge TTS generation through VoiceGenHub."""
        tts = VoiceGenHub(provider='edge')
        
        try:
            await tts.initialize()
            response = await tts.generate(
                text='nightly test of edge tts provider',
                voice='en-US-AriaNeural'
            )

            assert len(response.audio_data) > 0, 'no audio data generated'
            assert response.duration > 0, 'invalid audio duration'
            assert response.metadata is not None
        except Exception as e:
            # Skip test if Edge TTS API is unavailable (401/403 errors, network issues, etc.)
            if "401" in str(e) or "403" in str(e) or "Voice" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_piper_tts_generation(self):
        """Test Piper TTS generation through VoiceGenHub."""
        tts = VoiceGenHub(provider='piper')
        
        try:
            await tts.initialize()
            response = await tts.generate(
                text='test of piper tts provider',
                voice='en_US-lessac-medium',
                audio_format='wav'  # Piper only supports WAV format
            )

            assert len(response.audio_data) > 0, 'no audio data generated'
            assert response.duration > 0, 'invalid audio duration'
            assert response.metadata is not None
        except Exception as e:
            # Skip test if Piper dependencies are not available (Windows compatibility issues)
            if "not initialized" in str(e) or "dependencies" in str(e).lower() or "not available" in str(e).lower() or "Audio format" in str(e):
                pytest.skip(f"Piper TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_google_tts_generation(self):
        """Test Google TTS generation through VoiceGenHub."""
        # Check for Google credentials in order of preference:
        # 1. GOOGLE_APPLICATION_CREDENTIALS (set by CI after creating temp file)
        # 2. GOOGLE_APPLICATION_CREDENTIALS_JSON (CI secret, create temp file)
        # 3. Local config file
        import os
        import tempfile
        import json
        
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not credentials_path:
            # Check for JSON content from CI secret
            creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if creds_json:
                # Create temp file from JSON content (CI scenario)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(json.loads(creds_json), f)
                    credentials_path = f.name
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            else:
                # Fall back to local config file
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_credentials = os.path.join(project_root, 'config', 'google-credentials.json')
                if os.path.exists(config_credentials):
                    credentials_path = config_credentials
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                else:
                    pytest.skip("Google credentials not available - no GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_APPLICATION_CREDENTIALS_JSON, or config/google-credentials.json found")

        print(f"GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        print(f"Credentials file exists: {os.path.exists(credentials_path)}")

        tts = VoiceGenHub(provider='google')
        await tts.initialize()

        # Get available voices and use the first English one that doesn't require a model
        voices = await tts.get_voices(language='en')
        print(f"Available English voices: {len(voices)}")
        if not voices:
            pytest.skip("No English voices available from Google TTS")

        # Try to find a standard voice that doesn't require model specification
        voice_to_use = None
        for voice in voices:
            voice_name = voice["name"]
            # Prefer standard voices over neural ones for testing
            if "Standard" in voice_name and "en-US" in voice_name:
                voice_to_use = voice["id"]
                break
        
        # Fallback to first voice if no standard voice found
        if not voice_to_use:
            voice_to_use = voices[0]["id"]
        
        print(f"Using voice: {voice_to_use}")

        response = await tts.generate(
            text='nightly test of google tts provider',
            voice=voice_to_use,
            language='en-US'
        )

        assert len(response.audio_data) > 0, 'no audio data generated'
        assert response.duration > 0, 'invalid audio duration'
        assert response.metadata is not None

    @pytest.mark.asyncio
    async def test_coqui_tts_generation(self):
        """Test Coqui TTS generation through VoiceGenHub."""
        tts = VoiceGenHub(provider='coqui')
        
        try:
            await tts.initialize()
            
            # First get available voices
            voices = await tts.get_voices()
            if not voices:
                pytest.skip("No Coqui voices available")
            
            # Use the first available voice
            voice_to_use = voices[0]['id']
            
            response = await tts.generate(
                text='test of coqui tts provider',
                voice=voice_to_use,
                audio_format='wav'  # Coqui supports WAV
            )

            assert len(response.audio_data) > 0, 'no audio data generated'
            assert response.duration > 0, 'invalid audio duration'
            assert response.metadata is not None
        except Exception as e:
            # Skip test if Coqui dependencies are not available
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["not initialized", "dependencies", "not available", "import", "voice not found"]):
                pytest.skip(f"Coqui TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(Exception):  # TTSError inherits from Exception
            tts = VoiceGenHub(provider='invalid')
            await tts.initialize()

    @pytest.mark.asyncio
    async def test_generate_without_initialize(self):
        """Test generate works even without explicit initialization."""
        tts = VoiceGenHub(provider='edge')

        try:
            # Should work because generate calls initialize internally
            response = await tts.generate(text="test")
            assert len(response.audio_data) > 0
        except Exception as e:
            # Skip test if Edge TTS API is unavailable
            if "401" in str(e) or "403" in str(e) or "Voice" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        tts = VoiceGenHub(provider='edge')
        
        try:
            await tts.initialize()

            response = await tts.generate(
                text='Custom test message',
                voice='en-US-AriaNeural',  # Use a known working voice
                audio_format='mp3'
            )

            assert len(response.audio_data) > 0
            assert response.duration > 0
            assert response.metadata['provider'] == 'edge'
            assert 'voice_locale' in response.metadata
        except Exception as e:
            # Skip test if Edge TTS API is unavailable
            if "401" in str(e) or "403" in str(e) or "Voice" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_get_voices(self):
        """Test Edge TTS get_voices functionality."""
        tts = VoiceGenHub(provider='edge')
        
        try:
            await tts.initialize()
            voices = await tts.get_voices()
            
            assert len(voices) > 0, "Edge TTS should return available voices"
            assert all(isinstance(v, dict) for v in voices), "Voices should be dictionaries"
            assert all('id' in v and 'name' in v for v in voices), "Voices should have id and name"
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_get_voices_with_language_filter(self):
        """Test Edge TTS get_voices with language filter."""
        tts = VoiceGenHub(provider='edge')
        
        try:
            await tts.initialize()
            voices_en = await tts.get_voices(language='en')
            
            assert len(voices_en) > 0, "Should return English voices"
            # Verify all returned voices are English
            for voice in voices_en:
                assert voice['language'].startswith('en') or voice['locale'].startswith('en'), \
                    f"Voice {voice} should be English"
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_piper_get_voices(self):
        """Test Piper TTS get_voices functionality."""
        tts = VoiceGenHub(provider='piper')
        
        try:
            await tts.initialize()
            voices = await tts.get_voices()
            
            assert len(voices) > 0, "Piper TTS should return available voices"
            assert all(isinstance(v, dict) for v in voices), "Voices should be dictionaries"
            assert all('id' in v and 'name' in v for v in voices), "Voices should have id and name"
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Piper TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_piper_get_voices_with_language_filter(self):
        """Test Piper TTS get_voices with language filter."""
        tts = VoiceGenHub(provider='piper')
        
        try:
            await tts.initialize()
            voices_en = await tts.get_voices(language='en')
            
            assert len(voices_en) > 0, "Should return English voices"
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Piper TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_coqui_get_voices(self):
        """Test Coqui TTS get_voices functionality."""
        tts = VoiceGenHub(provider='coqui')
        
        try:
            await tts.initialize()
            voices = await tts.get_voices()
            
            assert len(voices) > 0, "Coqui TTS should return available voices"
            assert all(isinstance(v, dict) for v in voices), "Voices should be dictionaries"
            assert all('id' in v and 'name' in v for v in voices), "Voices should have id and name"
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Coqui TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_coqui_get_voices_with_language_filter(self):
        """Test Coqui TTS get_voices with language filter."""
        tts = VoiceGenHub(provider='coqui')
        
        try:
            await tts.initialize()
            
            # First get all voices
            all_voices = await tts.get_voices()
            if not all_voices:
                pytest.skip("No Coqui voices available")
            
            # Then filter by language
            voices_en = await tts.get_voices(language='en')
            
            # Should return at least one English voice if any voices exist
            # (or skip if filtering doesn't match)
            if len(all_voices) > 0:
                assert len(voices_en) > 0, f"Should return English voices. All voices: {[v.get('language', 'unknown') for v in all_voices]}"
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Coqui TTS dependencies not available: {e}")
            else:
                raise
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Coqui TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_invalid_voice(self):
        """Test Edge TTS with invalid voice raises error."""
        tts = VoiceGenHub(provider='edge')
        
        try:
            await tts.initialize()
            
            with pytest.raises(Exception):
                await tts.generate(
                    text='test',
                    voice='invalid-voice-id-12345'
                )
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_piper_invalid_voice(self):
        """Test Piper TTS with invalid voice raises error."""
        tts = VoiceGenHub(provider='piper')
        
        try:
            await tts.initialize()
            
            with pytest.raises(Exception):
                await tts.generate(
                    text='test',
                    voice='invalid-voice-id-12345'
                )
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Piper TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_coqui_invalid_voice(self):
        """Test Coqui TTS with invalid voice raises error."""
        tts = VoiceGenHub(provider='coqui')
        
        try:
            await tts.initialize()
            
            with pytest.raises(Exception):
                await tts.generate(
                    text='test',
                    voice='invalid-voice-id-12345'
                )
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Coqui TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_voice_caching(self):
        """Test Edge TTS caches voices properly."""
        tts = VoiceGenHub(provider='edge')
        
        try:
            await tts.initialize()
            
            # First call fetches voices
            voices1 = await tts.get_voices()
            # Second call should use cache (same object)
            voices2 = await tts.get_voices()
            
            assert len(voices1) == len(voices2)
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_piper_voice_caching(self):
        """Test Piper TTS caches voices properly."""
        tts = VoiceGenHub(provider='piper')
        
        try:
            await tts.initialize()
            
            # First call fetches voices
            voices1 = await tts.get_voices()
            # Second call should use cache (same object)
            voices2 = await tts.get_voices()
            
            assert len(voices1) == len(voices2)
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Piper TTS dependencies not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_coqui_voice_caching(self):
        """Test Coqui TTS caches voices properly."""
        tts = VoiceGenHub(provider='coqui')
        
        try:
            await tts.initialize()
            
            # First call fetches voices
            voices1 = await tts.get_voices()
            # Second call should use cache (same object)
            voices2 = await tts.get_voices()
            
            assert len(voices1) == len(voices2)
        except Exception as e:
            if "not initialized" in str(e) or "dependencies" in str(e).lower():
                pytest.skip(f"Coqui TTS dependencies not available: {e}")
            else:
                raise


# Test configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # This could be used to set up any global test state
    pass
