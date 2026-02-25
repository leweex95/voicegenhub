# Performance and Benchmarks

VoiceGenHub is designed for both local CPU-only systems and GPU-accelerated environments.

## Performance Comparison (Single Job)

| Provider | Quality (MOS) | Startup Time | Sequential (per req) | Async (3x parallel) | Model Size | Commercial |
|----------|---------------|--------------|---------------------|-------------------|------------|------------|
| **Edge TTS** | 3.8/5 | 4.9s | 3.2s | 2.5s | 0MB (cloud) | ✅ Free |
| **Kokoro** | 3.5/5 | 94s | 14.2s | 2.5s | 625MB | ✅ Apache 2.0 |
| **Bark** | 4.2/5 | 180s | 25-40s | 8-12s | 4GB | ✅ MIT |
| **Chatterbox** | 4.3/5 | 120s | 15-30s | 5-15s | 3.7GB | ✅ MIT |
| **ElevenLabs** | 4.5/5* | 2s | 3-5s | 2-3s | 0MB (cloud) | ⚠️ Paid API |

*ElevenLabs quality estimate based on reputation; not yet tested.*

## Concurrency Analysis (Chatterbox)

- **Memory Safety**: Chatterbox uses a **shared model instance** (3.6GB) across all threads — **no duplication**.
- **Performance**: ~2.8x speedup at 4 threads on CPU. Optimal thread count: **2-4 threads**.
- **Async Concurrency**: Safe to use 2-8 concurrent threads without OOM risk.

## [View Concurrency Plot](assets/concurrency_plot.html)
Interactive performance analysis showing speedup curves, memory usage, and timing breakdowns.

---
*For more details on Kaggle GPU benchmarks, see the remote GPU documentation.*
