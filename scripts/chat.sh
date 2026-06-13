#!/bin/bash
# Wrapper to make calling the mobile-llm fast and short
# Autodetects the project root relative to the script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Run the mobile_llm engine dynamically using the local gguf model
mobile_llm --backend llama.cpp --model "$PROJECT_ROOT/models/qwen1_5-0_5b-chat-q4_k_m.gguf" --prompt "$*"
