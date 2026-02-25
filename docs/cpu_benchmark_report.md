# CPU Generation Comparison Report: Qwen 3 vs. Chatterbox

This report compares the locally hosted performance of **Qwen 3 TTS** and **Chatterbox TTS** when running exclusively on **CPU**.

## Benchmark Methodology
- **Hardware:** Windows CPU (Local)
- **Environment:** Python 3.13, PyTorch 2.10.0 (Manual version, no flash-attn)
- **Input Phrases:**
  1. "Warm up." (Short)
  2. "The quick brown fox jumps over the lazy dog." (44 chars)
- **Metrics:** Initialization time, Generation time, Real-Time Factor (RTF).

## Results Summary

| Metric | Qwen 3 TTS (CPU) | Chatterbox TTS (CPU) |
|--------|------------------|----------------------|
| **Model Load/Init** | 25.04s | 37.42s |
| **Warm-up (2 words)** | ~464.0s (7.7 min) | ~80.0s (1.3 min) |
| **Text 1 (44 chars)** | > 720s (Timed out/Interrupted) | 98.71s |
| **Audio Duration** | ~1.5s (est) | 2.44s |
| **RTF (Estimated)** | **~300+** | **40.44** |

## Key Findings

### 1. Performance Gap
**Chatterbox is approximately 7-8 times faster than Qwen 3 on CPU.** While both are significantly slower than real-time (RTF > 1.0), Chatterbox is at least usable for very short snippets if patience is high, whereas Qwen 3 is unfeasibly slow for interactive use without GPU acceleration.

### 2. Qwen 3 Constraints
Qwen 3's performance suffers heavily on CPU due to:
- **Missing `flash-attn`:** The lack of flash attention forces a manual PyTorch implementation which is not optimized for CPU instruction sets.
- **Large Transformer Architecture:** Even the 0.6B version of Qwen 3 (CustomVoice) is heavy for single-threaded or unoptimized CPU inference.
- **RTF > 300:** Generating 1 second of audio takes over 5 minutes.

### 3. Chatterbox Advantages
Chatterbox performs better on CPU because:
- **Optimization:** The provider includes specific patches for CPU compatibility (float32 normalization, S3Tokenizer patches, etc.).
- **Architecture:** While still a transformer-based model (T3), it seems better suited for local CPU execution than the current Qwen 3 implementation in this environment.
- **RTF ~40:** Generating 1 second of audio takes about 40 seconds.

## Conclusion
If running locally on CPU is a requirement, **Chatterbox** is the only viable option between the two, though it remains quite slow. **Qwen 3** is essentially unusable on CPU without significant optimization or quantization (which was not evaluated here). For production CPU-only use, lighter providers (like Edge TTS or Kokoro) are recommended over these two heavier models.
