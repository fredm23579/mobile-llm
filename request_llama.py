import sys, json, urllib.request, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--chat", action="store_true")
parser.add_argument("--backend", default="llama.cpp", choices=["llama.cpp", "ollama", "huggingface"])
parser.add_argument("--model", default="llama3")
args, unknown = parser.parse_known_args()

prompt = sys.stdin.read()
is_chat = args.chat
backend = args.backend
model_name = args.model

if is_chat:
    system_prompt = "You are a helpful and intelligent AI assistant. Provide direct, clean answers."
    force_thought = False
    prefix = ""
    stop_tokens = ["<|im_end|>", "User:"]
else:
    system_prompt = "You are a highly capable autonomous OS agent. You MUST format your actions as:\nThought: <reason>\nAction: <tool>\nActionInput: <input>\n\nCRITICAL: STOP writing immediately after your ActionInput! Do NOT write a Final Answer until you receive an Observation!"
    force_thought = "Generating thought/action" in prompt or "Draft an execution" in prompt
    prefix = "Thought: " if force_thought else ""
    stop_tokens = ["<|im_end|>", "Observation:"]

formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{prefix}"
headers = {'Content-Type': 'application/json'}

if backend == "llama.cpp":
    url = "http://127.0.0.1:8080/completion"
    data = json.dumps({"prompt": formatted_prompt, "n_predict": 256, "temperature": 0.0, "stop": stop_tokens}).encode('utf-8')
elif backend == "ollama":
    url = "http://127.0.0.1:11434/api/generate"
    data = json.dumps({"model": model_name, "prompt": formatted_prompt, "stream": False, "raw": True, "options": {"temperature": 0.0, "stop": stop_tokens}}).encode('utf-8')
elif backend == "huggingface":
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    data = json.dumps({"inputs": formatted_prompt, "parameters": {"temperature": 0.001, "stop": stop_tokens, "return_full_text": False}}).encode('utf-8')
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token: headers["Authorization"] = f"Bearer {hf_token}"

req = urllib.request.Request(url, data=data, headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        
        if backend == "llama.cpp": text = result.get('content', '')
        elif backend == "ollama": text = result.get('response', '')
        elif backend == "huggingface": text = result[0].get('generated_text', '') if isinstance(result, list) else result.get('generated_text', '')
            
        if force_thought:
            text = "Thought: " + text
        print(text.strip())
except Exception as e:
    print(f"Error contacting API ({backend}): {e}")
