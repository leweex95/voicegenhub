# VoiceGenHub 🎙️

**Universal Text-to-Speech Library with Multi-Provider Support**

VoiceGenHub is a comprehensive, production-ready TTS (Text-to-Speech) library that provides a unified interface to multiple TTS providers. It features intelligent voice selection, advanced audio processing, robust error handling, and a powerful REST API.

## ✨ Features

### Core Capabilities
- 🔄 **Multi-Provider Support** - Edge TTS, Google Cloud TTS, Azure, and more
- 🎯 **Intelligent Voice Selection** - Automatic voice matching based on language, gender, emotion
- 🚀 **High Performance** - Async/await support, connection pooling, and caching
- 🔧 **Robust Error Handling** - Circuit breakers, automatic retries, fallback mechanisms
- 📡 **Streaming Support** - Real-time audio generation and streaming responses
- 🗄️ **Advanced Caching** - Multi-level caching with compression and TTL

### Audio Features
- 🎵 **Multiple Formats** - MP3, WAV, OGG, FLAC, AAC support
- 🎛️ **Audio Controls** - Speed, pitch, volume, and emotion control
- 📝 **SSML Support** - Full Speech Synthesis Markup Language support
- 🔧 **Audio Processing** - Format conversion, concatenation, effects

### Integration & APIs
- 🌐 **REST API** - Complete FastAPI-based REST interface
- 🖥️ **CLI Tools** - Command-line interface for batch processing
- 📚 **Comprehensive Docs** - OpenAPI/Swagger documentation
- 🐍 **Python Native** - Built for Python 3.11+ with type hints

## 🚀 Quick Start

### Installation

```bash
pip install voicegenhub
```

### Basic Usage

```python
import asyncio
from voicegenhub import VoiceGenHub

async def main():
    # Initialize TTS engine
    tts = VoiceGenHub()
    await tts.initialize()
    
    # Generate speech
    response = await tts.generate(
        text="Hello, world! This is VoiceGenHub.",
        language="en-US",
        voice="auto"  # Automatic voice selection
    )
    
    # Save audio file
    with open("speech.mp3", "wb") as f:
        f.write(response.audio_data)
    
    print(f"Generated {response.duration:.2f}s of audio using {response.voice_used}")

asyncio.run(main())
```

### Advanced Example

```python
import asyncio
from voicegenhub import VoiceGenHub
from voicegenhub.providers.base import AudioFormat

async def advanced_example():
    tts = VoiceGenHub(
        providers=["edge", "google"],  # Specify providers
        cache_enabled=True,
        fallback_enabled=True
    )
    await tts.initialize()
    
    # Generate with specific parameters
    response = await tts.generate(
        text="Welcome to our advanced TTS system!",
        voice="en-US-AriaNeural",  # Specific voice
        audio_format=AudioFormat.WAV,
        sample_rate=44100,
        speed=1.2,  # 20% faster
        emotion="excited"  # If supported by provider
    )
    
    # Stream audio generation
    with open("streaming_speech.mp3", "wb") as f:
        async for chunk in tts.generate_streaming(
            text="This is streamed audio generation.",
            language="en"
        ):
            f.write(chunk)

asyncio.run(advanced_example())
```

## 🛠️ CLI Usage

VoiceGenHub includes a powerful command-line interface:

```bash
# Generate speech from text
voicegenhub synthesize "Hello, world!" --voice en-US-AriaNeural --output hello.mp3

# List available voices
voicegenhub voices --language en --format table

# Check system health
voicegenhub health

# Start API server
voicegenhub serve

# Batch processing
voicegenhub batch texts.txt --output-dir ./audio --format wav
```

## 🌐 REST API

Start the API server:

```bash
voicegenhub serve
# or
python -m voicegenhub.api.main
```

API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### API Examples

```bash
# Generate speech
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from the API!",
    "voice": "auto",
    "language": "en-US",
    "format": "mp3"
  }'

# Get available voices
curl "http://localhost:8000/voices?language=en"

# Health check
curl "http://localhost:8000/health"
```

## 🔧 Configuration

### Environment Variables

```bash
# General settings
VOICEGENHUB_DEBUG=true
VOICEGENHUB_LOG_LEVEL=INFO
VOICEGENHUB_CACHE_ENABLED=true

# API settings
VOICEGENHUB_API_HOST=0.0.0.0
VOICEGENHUB_API_PORT=8000

# Provider credentials
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
AZURE_SUBSCRIPTION_KEY=your_azure_key
AZURE_REGION=eastus
```

### Configuration File

Create a `config.yaml` file:

```yaml
debug: false
log_level: INFO
cache_enabled: true
default_format: mp3
default_sample_rate: 22050

providers:
  edge:
    enabled: true
    rate_limit_delay: 0.1
  
  google:
    enabled: true
    project_id: your-project-id
    
  azure:
    enabled: true
    region: eastus

api:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["*"]
```

## 🏗️ Architecture

VoiceGenHub follows a modular architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │    │   CLI Interface  │    │  Python Library │
└─────────┬───────┘    └────────┬─────────┘    └─────────┬───────┘
          │                     │                        │
          └─────────────────────┼────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    VoiceGenHub Core   │
                    │      (Engine)         │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Provider Factory    │
                    │   & Registry          │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
      ┌─────────▼─────────┐ ┌──▼──┐ ┌─────────▼─────────┐
      │   Edge TTS        │ │ ... │ │  Google Cloud TTS │
      │   Provider        │ │     │ │  Provider         │
      └───────────────────┘ └─────┘ └───────────────────┘
```

### Key Components

- **VoiceGenHub Engine**: Main orchestration layer
- **Provider Factory**: Dynamic provider registration and instantiation
- **Voice Selector**: Intelligent voice selection with multiple strategies
- **Audio Pipeline**: Format conversion, effects, and processing
- **Cache System**: Multi-level caching with TTL and compression
- **Error Handler**: Circuit breakers, retries, and fallback mechanisms

## 🎯 Voice Selection

VoiceGenHub provides sophisticated voice selection capabilities:

```python
from voicegenhub.core.voice import VoiceSelectionCriteria, VoiceSelectionStrategy

# Define selection criteria
criteria = VoiceSelectionCriteria(
    language="en-US",
    gender=VoiceGender.FEMALE,
    age_group="adult",
    emotion="cheerful",
    preferred_providers=["edge", "google"],
    strategy=VoiceSelectionStrategy.BEST_MATCH
)

# Select voice
voice = await voice_selector.select_voice(criteria)
```

### Selection Strategies

- **BEST_MATCH**: Comprehensive scoring based on all criteria
- **QUALITY_FIRST**: Prioritize highest quality voices
- **SPEED_FIRST**: Optimize for fastest generation
- **RANDOM**: Random selection from matching voices
- **ROUND_ROBIN**: Cycle through available voices

## 🔌 Provider Support

### Currently Supported

| Provider | Status | Features |
|----------|--------|----------|
| Microsoft Edge TTS | ✅ Complete | Neural voices, SSML, Free |
| Google Cloud TTS | 🚧 In Progress | WaveNet, Custom voices, Premium |
| Azure Cognitive Services | 📋 Planned | Neural voices, Custom voices |
| AWS Polly | 📋 Planned | Neural voices, Lexicons |
| ElevenLabs | 📋 Planned | AI voices, Voice cloning |

### Adding Custom Providers

```python
from voicegenhub.providers.base import TTSProvider

class CustomTTSProvider(TTSProvider):
    @property
    def provider_id(self) -> str:
        return "custom"
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        # Implementation here
        pass

# Register provider
from voicegenhub.providers.factory import register_provider
register_provider(CustomTTSProvider)
```

## 📊 Performance & Monitoring

### Caching

VoiceGenHub implements intelligent caching:

- **Memory Cache**: Fast access for frequently used audio
- **Disk Cache**: Persistent storage with compression
- **Distributed Cache**: Redis support for multiple instances
- **Smart Invalidation**: Content-based hashing and TTL

### Monitoring

```python
# Health monitoring
health = await tts.health_check()
print(f"System health: {health['overall_health']}")
print(f"Cache hit rate: {health['cache_stats']['hit_rate']}")

# Performance metrics
metrics = await tts.get_metrics()
print(f"Average response time: {metrics['avg_response_time']}ms")
print(f"Total requests: {metrics['total_requests']}")
```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=voicegenhub tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/voicegenhub.git
cd voicegenhub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Edge TTS team for the excellent free TTS service
- Google Cloud team for their high-quality TTS APIs
- The Python community for amazing open-source libraries
- Contributors and users who make this project better

## 📞 Support

- 📚 **Documentation**: [https://voicegenhub.readthedocs.io](https://voicegenhub.readthedocs.io)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/voicegenhub/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/voicegenhub/discussions)
- 📧 **Email**: csibi.levente14@gmail.com

---

**Made with ❤️ by the VoiceGenHub team**