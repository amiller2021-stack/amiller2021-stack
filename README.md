### Aaron Miller
**Small-Model Post-Training · Domain-Specialized Models · Adapter Merging**

I focus on post-training smaller, specialized models for high-precision disciplines—particularly **law, finance, and banking**—where structured document reasoning, citation fidelity, and verifiable numeric/date accuracy matter more than generic chat fluency. Most of my prior experimentation has run on private local and cloud environments; this profile hosts clean, standalone PyTorch utilities, calibration tools, and deterministic verifiers extracted from that work.

#### Featured Repositories
* **[`anvil-renyi-lora`](https://github.com/amiller2021-stack/anvil-renyi-lora)** — Standalone PyTorch modules for 3-step Newton–Schulz polar LoRA updates, SVD-free Rényi-2 effective-rank regularization (`r × r` Gram formulation), and FP32 weight-soup (`W_soup`) multi-trajectory merging.
* **[`hidden-state-aux-calibrator`](https://github.com/amiller2021-stack/hidden-state-aux-calibrator)** — Hidden-state extraction and lightweight auxiliary predictor calibration against locked checkpoints for reasoning-halting control, speculative transition scoring, and intermediate-layer probing.
* **[`rlvr-action-verifier`](https://github.com/amiller2021-stack/rlvr-action-verifier)** — Deterministic action-log vs. transcript parity auditor, mechanical tool-output compactor, and exact arithmetic/date verifiers built for RLVR and post-training evaluation loops.

#### Core Focus
* **Domain-Specialized Small Models (Law, Finance & Banking):** Curating weighted, multi-priority domain corpora (~200M tokens) and running continual post-training on compact Qwen-family models (1.5B–2B class) for dense document extraction, contract/regulatory reasoning, and financial analysis.
* **LoRA Optimization & Modularity:** Training independent LoRA trajectories (`r=64` base adaptation and `r=32` modular domain adapters), preventing low-rank spectral collapse via Rényi-2 regularization, and merging checkpoints in FP32 to retain base capabilities.
* **Inference Efficiency & Reasoning Control:** Calibrating auxiliary heads directly against frozen hidden states for adaptive halting and speculative scoring, alongside imatrix-calibrated post-training quantization.
* **Verifiable Evals & RLVR:** Pairing small models with deterministic execution-log audits and strict numeric/temporal oracles to eliminate hallucinated tool claims and calculation drift.

#### Exploring on `whitecircle/halo`
Adapting this small-model post-training workflow to Halo's distributed stack—specifically targeting:
* **SFT + Multi-Token Prediction (MTP) heads** to pair domain accuracy with faster decoding
* **Grouped LoRA on fused MoE experts** with multi-trajectory merging
* **Async GRPO** driven by deterministic action-log and domain-specific verifiers

#### Stack
`PyTorch` · `Transformers` · `PEFT` · `Halo` · `llama.cpp` · `DuckDB / SQLite`

#### Contact
* **Discord:** `@amiller424`
