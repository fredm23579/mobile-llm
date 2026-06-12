# 🚀 MobileLLM Launch Kit

To help you get the word out, I've drafted high-impact launch templates specifically tailored to the culture of each major platform. Simply copy, paste, and post!

---

## 🟥 Hacker News (Show HN)

**Title:** Show HN: MobileLLM - An O(N) Linear-Time LLM Inference Engine for Android Termux
**Body:**
Hey HN,

I built MobileLLM: a zero-Python, bare-metal C++17 inference engine designed to bypass the $O(N^2)$ memory constraints of traditional Transformers and run multi-gigabyte models natively on mobile CPUs (specifically Android Termux).

Traditional attention rapidly exhausts mobile RAM. To fix this, I implemented a Linear Recurrent State-Space model (similar to RWKV/Mamba) that guarantees a strict $O(N)$ inference speed and $O(1)$ memory footprint per token. 

To maximize speed, critical inner-loop mathematical decay functions are passed via raw memory pointers directly to `!DIR$ SIMD` optimized Fortran 2003 binaries. I also wrote a custom Eigen3 implementation ("TurboQuant") to squash 32-bit float vector spaces into strictly bounded `[-127, 127]` INT8 arrays, slashing KV-cache constraints.

The engine isn't just a chatbot—it features an infinite Karpathy-style AutoResearch loop natively integrated with a 13-tool Termux OS bridge. The LLM can autonomously read/write files, execute bash, fetch raw HTML, and parse `.json`/`.bin` data without any Python wrappers.

I'd love for you to try it out or critique the Fortran SIMD architecture.

Repo: https://github.com/fredm23579/mobile-llm

---

## 🟧 Reddit (r/LocalLLaMA & r/MachineLearning)

**Title:** I built a bare-metal C++ / Fortran LLM engine that runs O(N) linear-time inference directly on Android Termux.

**Post:**
Hey everyone,

We all know the pain of running large models on constrained memory: context windows eat up RAM quadratically $O(N^2)$. I wanted to see if I could push a mobile CPU to its absolute physical limits, so I wrote **MobileLLM**.

It’s an open-source, zero-Python inference engine built entirely on C++17 and Fortran. 

**How it works:**
*   **O(N) Complexity:** It abandons standard attention for a linear state-space architecture. Memory footprint per token is constant $O(1)$.
*   **Fortran SIMD Core:** I bypassed PyTorch C++ abstraction overhead by passing raw memory pointers directly into a bare-metal Fortran 2003 routine for the recurrent decay math.
*   **TurboQuant INT8:** Uses Eigen3 orthogonal rotations to mathematically bound the KV-cache to INT8 without losing structural integrity.
*   **Native AutoResearch:** It doesn't rely on LangChain. The Karpathy-style autonomous loop is written directly in C++ and connects the LLM to a 13-tool OS arsenal (it can run native `grep`, native `<cstdio>` file unlinking, and headless web scraping dynamically).

You can compile it straight from Termux using `cmake` and `make`.

Check it out here: [https://github.com/fredm23579/mobile-llm](https://github.com/fredm23579/mobile-llm)

Would love your feedback on the quantization approach!

---

## 🩵 Twitter / X (Thread)

**Tweet 1:**
Traditional LLMs crash on mobile because memory scales quadratically O(N²). 
I just open-sourced MobileLLM: A linear-time O(N) inference engine built on pure C++ and Fortran that runs multi-gigabyte models natively on Android Termux. 📱🧠
GitHub link below ⬇️

**Tweet 2:**
How is it so fast? Zero Python. 🐍❌
Critical recurrent math is piped directly into a bare-metal Fortran SIMD Core. Meanwhile, an Eigen3 'TurboQuant' engine squashes the KV-cache into strictly bounded INT8 arrays to preserve precious mobile RAM.

**Tweet 3:**
It’s not just a chatbot. MobileLLM natively implements an infinite Karpathy-style AutoResearch loop. 
It possesses a 13-tool OS arsenal, allowing the LLM to autonomously run Bash commands, scrape raw HTML, and parse binary files directly on your device.

**Tweet 4:**
If you're interested in pushing edge AI to its absolute physical limits without bloated wrappers, check out the source code:
[https://github.com/fredm23579/mobile-llm](https://github.com/fredm23579/mobile-llm)
Star the repo if you like it! 🌟

---

## 💡 Discord / Slack (Quick Drop)

"Hey guys, I just open-sourced a project I've been working on: **MobileLLM**. It's a bare-metal C++17/Fortran inference engine that runs linear O(N) state-space models directly on Android Termux. It completely bypasses Python, uses custom INT8 Eigen quantization, and features a natively compiled AutoResearch agent loop with 13 OS-level tools. Would love for you guys to check out the repo and tear apart the architecture! https://github.com/fredm23579/mobile-llm"
