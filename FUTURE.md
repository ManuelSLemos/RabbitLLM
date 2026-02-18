# 🐇 RabbitLLM

> **RabbitLLM auto-fits LLMs to your hardware.**
> One command. Smart memory. Clean logs.

RabbitLLM is a minimal, opinionated runtime that automatically adapts large language models to your GPU or CPU.
No complex flags. No manual tuning. Just run.

---

## ✨ Why RabbitLLM?

Running LLMs on consumer GPUs is messy:

* Out-of-memory crashes
* Endless configuration flags
* Unclear performance tradeoffs
* Hidden memory behavior

RabbitLLM solves this with one core idea:

> **The runtime decides the best execution strategy for your hardware.**

---

## 🧠 Core Philosophy

* **Minimal surface area**
* **Auto-tuning by default**
* **Predictable behavior**
* **Transparent decisions**
* **Clean, readable internals**

RabbitLLM does not try to be:

* A full agent framework
* A UI platform
* A benchmarking circus
* A “fastest on H100” engine

It is built for **real hardware** — like RTX 4060 Ti, 8–16GB GPUs, and edge devices.

---

## 🚀 Quick Start

### Install

```bash
pip install rabbitllm
```

### Pull a model

```bash
rabbit pull qwen2.5:7b
```

### Run locally

```bash
rabbit run qwen2.5:7b
```

### Serve OpenAI-compatible API

```bash
rabbit serve qwen2.5:7b
```

That’s it.

---

## ⚙️ Auto Strategy (No Flags Needed)

RabbitLLM automatically selects one of three internal execution profiles:

| Profile  | Goal                    |
| -------- | ----------------------- |
| balanced | Default smart behavior  |
| fast     | More VRAM, higher speed |
| tight    | Guaranteed to fit       |

Example log:

```
plan=tight
reason="7.9GB VRAM available, model requires 10.6GB"
strategy="layer_stream + kv_offload + reduced_batch"
```

You always know what Rabbit decided — and why.

---

## 🏗 Architecture

RabbitLLM is intentionally small.

```
rabbit/
 ├── registry/      # Model resolution & cache
 ├── planner/       # Auto strategy engine
 ├── executor/      # Adaptive memory execution
 ├── api/           # OpenAI-compatible server
 └── tools/         # doctor & bench
```

### 1️⃣ Model Registry

* HuggingFace-based resolution
* Deterministic caching
* Stable model naming (`qwen2.5:7b`)

### 2️⃣ Planner

* Detects VRAM in real time
* Calculates memory requirements
* Selects precision, offload policy, KV strategy
* Produces a deterministic `ExecutionPlan`

### 3️⃣ Executor

* Layer streaming (AirLLM-inspired)
* Dynamic CPU/GPU offload
* Adaptive KV cache
* Streaming token generation
* Cancellation support

### 4️⃣ API Server

Minimal OpenAI-compatible endpoints:

* `GET /v1/models`
* `POST /v1/chat/completions` (with streaming)

---

## 🧪 Benchmarking

Rabbit includes reproducible benchmarks:

```bash
rabbit bench qwen2.5:7b
```

Metrics:

* Tokens/sec
* Time-to-first-token
* Peak VRAM
* Stability (OOM detection)

Designed for **real GPUs**, not datacenter hardware.

---

## 🩺 Diagnostics

```bash
rabbit doctor
```

Checks:

* CUDA availability
* Driver compatibility
* VRAM health
* Torch/CUDA versions
* Recommended profile

---

## 🔍 What Makes RabbitLLM Different?

* Adaptive memory management instead of static configs
* Layer-by-layer execution for low VRAM environments
* Clear execution reasoning
* Minimal CLI
* OpenAI-compatible out of the box
* Designed for consumer GPUs

RabbitLLM optimizes for **survival + elegance**, not synthetic throughput numbers.

---

## 📦 Configuration

Optional `rabbit.toml`:

```toml
device = "auto"
profile = "balanced"
max_vram_percent = 90
cache_dir = "~/.rabbit"
```

If you don’t create this file, Rabbit still works.

---

## 🛣 Roadmap

* Adaptive KV compression
* Prefetch optimization
* Multi-stream scheduling
* ROCm support
* Modular backend interface

---

## 🤝 Contributing

RabbitLLM values:

* Clean PRs
* Measurable improvements
* Simplicity over feature explosion
* Transparent benchmarking

Before submitting a PR:

* Run `rabbit bench`
* Include memory impact
* Keep the CLI minimal

---

## 🐇 Design Goal

RabbitLLM should feel like:

> “It just works — and I understand why.”

---

## License

MIT
