# 🧠 Training MobileLLM

MobileLLM is designed for **inference** on edge devices. However, because it uses a custom $O(N)$ linear state-space architecture, you cannot use standard LLaMA weights. You must train it from scratch on a GPU cluster.

## Why Can't We Train It On Android/Termux?
Training an LLM requires executing the forward pass, calculating the loss against terabytes of text data, computing the mathematical gradients (backpropagation), and updating the weights. Doing this for even a 1-Billion parameter model requires hundreds of A100 GPUs running for weeks. If we attempted to train this on an Android CPU or a basic container, it would literally take thousands of years to complete 1 epoch.

## How Many Parameters Does It Have?
Currently, the C++ template in `main.cpp` is hardcoded to a tiny scale for compilation testing:
*   `d_model = 256`
*   `vocab_size = 32000`
*   **Total Parameters:** ~8.1 Million.

To scale this to a production-ready model like TinyLlama (1.1 Billion parameters), you simply change the code to `d_model = 2048` and add stacked layers. 

## How To Train It (GPU Cluster Required)

To train this, you need to spin up a GPU instance (AWS, GCP, RunPod) and write a Python PyTorch script that perfectly mirrors the C++ mathematical architecture we built. Once trained, you export it to `.gguf` and load it back into Termux.

Here is the exact PyTorch architectural equivalent you would use to train the model:

```python
import torch
import torch.nn as nn

class LinearStateSpaceLLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=2048, decay=0.9):
        super().__init__()
        self.d_model = d_model
        self.decay = decay
        
        # Word Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Output Projection Matrix (W_out)
        self.W_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Initialize the O(N) linear hidden state
        hidden_state = torch.zeros(batch_size, self.d_model).to(input_ids.device)
        logits = []

        # Autoregressive forward pass
        for t in range(seq_len):
            token_emb = self.embedding(input_ids[:, t])
            
            # The mathematical equivalent of our Fortran SIMD decay function
            hidden_state = (self.decay * hidden_state) + token_emb
            
            # Project to vocabulary size
            token_logits = self.W_out(hidden_state)
            logits.append(token_logits.unsqueeze(1))

        return torch.cat(logits, dim=1)

# Training Loop Concept
model = LinearStateSpaceLLM().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# You would feed terabytes of tokenized Wikipedia/WebText into this loop
for batch in dataloader:
    inputs, targets = batch
    predictions = model(inputs.cuda())
    
    loss = torch.nn.functional.cross_entropy(predictions.view(-1, 32000), targets.view(-1).cuda())
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
# Finally, export to GGUF format
# save_as_gguf(model.state_dict(), "model.gguf")
```
