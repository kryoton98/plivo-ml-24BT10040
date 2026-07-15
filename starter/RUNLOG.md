# Run Log - LLM Speedrun

## Run 1: Baseline Architecture & Tokenizer
* **Hypothesis:** The out-of-the-box plain PyTorch baseline tokenizer is highly inefficient for mixed English-Hindi text because Devanagari characters fragment into 3 tokens per character, wasting valuable parameter capacity and shortening the context window.
* **What Changed:** Evaluated the untrained random initialization baseline to set a baseline metric.
* **Dev bpb Before/After:** 
  * Before: N/A
  * After: 3.6125 (Untrained random baseline)
* **Conclusion:** The compression ratio must be improved immediately via an optimized tokenizer to give the core network more context per sequence window.

## Run 2: High-Efficiency Llama-Class Sprint
* **Hypothesis:** By implementing a custom Byte Pair Encoding (BPE) tokenizer alongside an aggressive Llama-class parameter-efficient transformer architecture (Weight Tying, Multi-Query Attention, SwiGLU, RMSNorm, and RoPE), we can maximize representational capacity within the strict 2M parameter cap and 2,000-step budget.
* **What Changed:** 
  * Replaced the byte tokenizer with a 1024-vocab Regex-based BPE tokenizer (768 merges, achieving 2.66x compression).
  * Overhauled `model.py` to use a 6-layer MQA Llama-style network (1,979,648 parameters).
  * Rebuilt `train.py` with AdamW (betas 0.9, 0.95), Cosine Annealing schedule with 10% linear warmup, gradient clipping at 1.0, and a simulated effective batch size of 64 via Gradient Accumulation.
* **Dev bpb Before/After:** 
  * Before: 3.6125
  * After: 1.6887
* **Conclusion:** The massive 2.66x compression ratio allowed the model to process more text per sequence chunk, while weight-tying and MQA safely packed a high-capacity 6-layer model completely under the strict 2,000,000 parameter threshold.