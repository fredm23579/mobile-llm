import sys, json, urllib.request

prompt = sys.stdin.read()

is_chat = "--chat" in sys.argv

if is_chat:
    system_prompt = "You are a helpful and intelligent AI assistant. Provide direct, clean answers."
    force_thought = False
    prefix = ""
    stop_tokens = ["<|im_end|>", "User:"]
else:
    system_prompt = "You are a highly capable autonomous OS agent. You MUST format your actions as:\nThought: <reason>\nAction: <tool>\nActionInput: <input>\n\nWhen completely finished with your task, output:\nFinal Answer: <answer>"
    force_thought = "Generating thought/action" in prompt or "Draft an execution" in prompt
    prefix = "Thought: " if force_thought else ""
    stop_tokens = ["<|im_end|>", "Observation:"]

data = json.dumps({
    "prompt": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{prefix}",
    "n_predict": 256,
    "temperature": 0.0,
    "stop": stop_tokens
}).encode('utf-8')

req = urllib.request.Request("http://127.0.0.1:8080/completion", data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        text = result.get('content', '')
        if force_thought:
            text = "Thought: " + text
        print(text.strip())
except Exception as e:
    print(f"Error contacting Llama Server: {e}")
