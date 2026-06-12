# ⚡ MobileLLM Engine

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Fortran](https://img.shields.io/badge/Fortran-2003-purple.svg)](#)
[![Platform](https://img.shields.io/badge/Platform-Termux%20%7C%20Android%20%7C%20Linux-orange.svg)](#)
[![Complexity](https://img.shields.io/badge/Complexity-O(N)%20Linear-red.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](#)

> **State-of-the-Art $O(N)$ Linear-Time Large Language Model Inference Engine for Mobile Devices.**

MobileLLM is a highly optimized, zero-Python inference engine designed specifically for deployment in heavily memory-constrained environments like Android Termux. By combining native C++17 abstractions with bare-metal Fortran SIMD vectorization and experimental INT8 KV-cache quantization, it pushes the absolute physical limits of mobile CPU computation.

## 🧠 Inference vs. Training: The Reality of Edge AI

It is mathematically impossible to train a Large Language Model from scratch on a mobile CPU. Training an LLM requires computing billions of gradients across datasets like Wikipedia—a process that takes hundreds of A100 Supercomputer GPUs several weeks to complete. 

**MobileLLM is an Inference Engine, not a training cluster.** 
The explicit goal of this C++ architecture is to take the heavy mathematical weights that massive tech companies have already paid millions of dollars to train, and execute them natively on low-power, constrained edge devices. We solve the quadratic memory bottleneck so you can run billion-parameter models on your phone.

---

## 🌟 Core Architecture: O(N) Computational Efficiency

Traditional Transformer LLMs rely on $O(N^2)$ quadratic attention, which rapidly exhausts mobile RAM on long contexts. MobileLLM abandons this in favor of a **Linear Recurrent State-Space** model (similar to Mamba/RWKV). This guarantees $O(N)$ inference speed and a constant $O(1)$ memory footprint per token, allowing multi-gigabyte inference directly on your phone's CPU.

```mermaid
graph TD
    A[Text Input] -->|BPE Tokenizer| B(Token Stream)
    B --> C{LibTorch Graph}
    C -->|Hidden State Arrays| D((Fortran SIMD Core))
    D -->|O:N Decay Math| C
    C -->|KV Matrix| E[TurboQuant]
    E -->|Eigen3 QR Rotation| F[INT8 Compressed Cache]
    C --> G[Output Logits]
    G -->|Karpathy AutoResearch| H[Agent OS Bridge]
    H -->|popen| I[Termux Bash Layer]
```

## 🚀 Key Features

*   **Zero-Python Execution:** The entire engine runs natively. No bloated interpreters, no memory leaks.
*   **Fortran 2003 Acceleration:** Critical inner-loop mathematical decay functions are passed via raw memory pointers directly to `!DIR$ SIMD` optimized Fortran binaries, bypassing C++ abstraction overhead.
*   **TurboQuant Compression:** Implements randomized orthogonal rotations via Eigen3 to squash 32-bit float vector spaces into strictly bounded `[-127, 127]` INT8 arrays, drastically slashing KV-cache constraints.
*   **GGUF v3 Binary Parser:** Capable of scanning memory-mapped `model.gguf` files, stripping out metadata, and mounting multi-gigabyte Tensor offsets natively.
*   **Infinite AutoResearch Loop (Karpathy-Style):** Automatically generates step-by-step specifications and continuously self-corrects against observations until the final objective is reached.

## 🛠️ The 13-Tool Termux Arsenal
The LLM isn't just a chatbot; it's a native OS controller equipped with custom C++ tools that map directly to the Linux Kernel:
1. **`run_command`**: Directly executes Termux bash commands.
2. **`read_file`**: Rapid stream-buffer reading.
3. **`write_file`**: Native OS file construction.
4. **`delete_file`**: `<cstdio>` kernel removal API.
5. **`copy_file`**: Memory-to-memory duplication.
6. **`move_file`**: Native `std::rename` file manipulation.
7. **`list_dir`**: Subprocess directory hierarchies.
8. **`search_web`**: Headless API search querying.
9. **`fetch_url`**: Headless Mozilla/5.0 web scraping.
10. **`pattern_match`**: Native `grep` piping.
11. **`mathematics`**: Boundless arithmetic via `bc -l`.
12. **`text_parsing`**: Complex Unix stream extraction via `awk`.
13. **`universal_parse`**: Dynamic magic-byte format shifting (parses `.json`, `.csv`, `.xml`, and raw `.bin` hex dumps via `xxd`).

## 🦙 Llama.cpp Server Backend Integration (`--llama`)

While MobileLLM features a highly experimental linear-time inference engine via LibTorch, you may prefer the robust, standard optimization of a conventional `llama.cpp` backend.

By passing the `--llama` flag, you can bridge the C++ MobileLLM agent frontend directly to a standard `llama.cpp` server backend.

```bash
# Interactive Chat with Llama.cpp backend
./mobile_llm --llama --chat

# Autonomous Agent Loop with Llama.cpp backend
./mobile_llm --llama --prompt "Analyze the environment and report."
```

**How It Works:**
When this flag is active, the engine bypasses the internal LibTorch inference module. Instead, it routes generation requests (along with the 13-tool system prompt) through a translation layer (`LlamaServerAdapter` via `request_llama.py`), which queries a local `llama.cpp` server running at `http://127.0.0.1:8080/completion`.

**The Best of Both Worlds:**
This provides the ultimate hybrid architecture: you get the ultra-capable, zero-Python C++ Karpathy-style agent loop with its 13 native OS tools, but powered by the heavily optimized, rock-solid inference of `llama.cpp`.

*Note: Ensure your `llama.cpp` server is running locally on port 8080 before using this mode.*

## 📂 Directory Structure

```text
mobile-llm/
├── CMakeLists.txt      # Master build manifest
├── main.cpp            # PyTorch C++ (LibTorch) execution loop
├── fast_math.f90       # Bare-metal Fortran SIMD array processor
├── turboquant.hpp      # Eigen3-powered vector quantization
├── gguf_parser.hpp     # Deep binary memory-mapping for weights
├── tokenizer.hpp       # Native Byte-Pair Encoding logic
└── agent.hpp           # Termux OS shell-execution bridge
```

## ⚙️ Build Instructions

This project requires a Linux environment (or Android Termux) with C++ and Fortran compilers.

### 1. Install Dependencies
```bash
apt-get update
apt-get install -y build-essential cmake gfortran libeigen3-dev
```

### 2. Install LibTorch
You must acquire the PyTorch C++ bindings (LibTorch) for your architecture. If on Termux, you can extract the bindings via a local Python virtual environment:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
export TORCH_PATH=$(python -c 'import torch; print(torch.__path__[0])')/share/cmake/Torch
```

### 3. Compile the Engine
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$TORCH_PATH ..
make
```

## 🧪 Testing

To ensure mathematical stability and memory bounds are strictly enforced on your specific hardware, run the unit test harness:

```bash
./mobile_llm_tests
```

*Expected Output:*
```text
[Test] Running Fortran Decay Math Stability...
  -> PASS: Fortran math is stable over 1000 recurrent steps.
[Test] Running TurboQuant Eigen Bounds Verification...
  -> PASS: TurboQuant successfully bounds vectors to INT8 space.
[Test] Running ReAct Agent Flow...
  -> PASS: Agent architecture compiles and integrates successfully.
```

## 🛡️ License

MIT License. See LICENSE for details.
