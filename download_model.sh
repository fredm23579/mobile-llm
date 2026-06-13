#!/bin/bash
# Defines safe, local defaults for model storage
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODEL_DIR"

# Example: Downloading a very small (0.5B), fast model for testing
MODEL_URL="https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf"
OUTPUT_FILE="$MODEL_DIR/qwen1_5-0_5b-chat-q4_k_m.gguf"

echo "Downloading Qwen1.5-0.5B-Chat GGUF from Hugging Face..."
wget -q --show-progress -O "$OUTPUT_FILE" "$MODEL_URL"

echo "Download complete! Model saved to: $OUTPUT_FILE"
echo "You can now run: mobile_llm -m $OUTPUT_FILE -p \"Hello, world!\""
