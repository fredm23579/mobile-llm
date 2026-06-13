# ⚡ MobileLLM Usage & CLI Guide

This guide details how to use `mobilellm` natively as a CLI tool within your terminal environment.

## 🚀 Environment Setup

The binary has been symlinked directly into your system `$PATH` (`/usr/local/bin/mobilellm`). You can now invoke it from anywhere on your device!

## ⚙️ Command-Line Syntax

```bash
mobilellm [--prompt "Your agentic goal"] [--model "/path/to/model.gguf"] [--backend "framework_name"] [--chat]
```

### Arguments:
*   `--prompt`: (String) The high-level objective you want the AutoResearch agent to accomplish.
*   `--model`: (String) The path to the physical GGUF weights on your device. Defaults to `model.gguf`.
*   `--backend`: (String) The underlying engine framework to process the weights. Options:
    *   `native-linear`: (Default) Runs the O(N) Sub-Polynomial C++ Linear State architecture.
    *   `native-transformer`: Runs standard O(N^2) Transformer Attention in native C++.
    *   `native-mamba`: Runs the State Space Model (SSM) continuous-time architecture.
    *   `llama.cpp`: Hooks directly into the llama.cpp C headers for standard LLaMA models.
*   `--chat`: (Flag) Bypasses the ReAct loop and launches a standard interactive terminal chat session.

## 🤖 Running an Agentic Workflow

Unlike traditional LLMs that just spit out text, MobileLLM parses your prompt and immediately enters the **Karpathy AutoResearch Loop**. 

### Example 1: System Analysis
```bash
mobilellm --prompt "Find the top 3 largest directories in /var and save the output to /root/largest_dirs.txt"
```
**What happens under the hood?**
1. The engine drafts a precise specification to achieve the goal.
2. It uses the `run_command` tool to execute `du -sh /var/* | sort -rh | head -n 3`.
3. It captures the stdout and uses the `write_file` tool to save the results to `/root/largest_dirs.txt`.
4. It validates the file using `read_file` before breaking the loop and returning the `Final Answer:`.

### Example 2: Mathematical Code Extraction
```bash
mobilellm --prompt "Fetch the raw HTML from google.com, extract all links, and calculate how many bytes the file takes up."
```
**What happens under the hood?**
1. Uses `fetch_url` to rip the raw HTML into memory.
2. Uses `pattern_match` to grep the raw `href=` tags.
3. Uses `mathematics` to calculate `wc -c` data constraints natively using the `bc` Kernel API.

## 🛠️ Modifying the Agent's Tools

If you want the agent to learn a completely new trick, you do not need to modify PyTorch or Python. You just write raw C++:

1. Open `agent.hpp`.
2. Scroll to the `run_autoresearch_loop` parsing blocks.
3. Add a new `else if (action == "my_new_tool") { ... }` executing your custom `popen()` logic.
4. Run `make` to compile it down to the native SIMD binary.
