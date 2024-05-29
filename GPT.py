import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

# Create Vocabulary
path = './data/HarryPotterPreprocessed.txt'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))

# Tokenization
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Hyperparameters
VOCAB_SIZE = len(chars)
EMBEDDING_SIZE = 384
CONTEXT_SIZE = 256
BATCH_SIZE = 64
MAX_STEPS = 5000
LEARNING_RATE = 3E-4
BLOCK_COUNT = 6
NUM_HEADS = 6
DROPOUT = 0.2
HEAD_SIZE = 64 # How big Query, Key and Value matrices are
device = 'cuda' if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL = 500
EVAL_LOSS_BATCHES = 200

# split data into train & validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(data.shape[0] * 0.9)
train_data, val_data = data[:n], data[n:]

# Loader that returns a batch
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - CONTEXT_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# calculate mean loss for {EVAL_LOSS_BATCHES}x batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_LOSS_BATCHES, device=device)
        for i in tqdm(range(EVAL_LOSS_BATCHES)):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# a Single Head of Self-Attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.key = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        
        self.dropout = nn.Dropout(DROPOUT)
        # since it's not a parameter of the model => register as buffer
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
    
    def forward(self, x):
        n_batch, n_context, n_emb = x.shape

        q, k, v = self.query(x), self.key(x), self.value(x)

        # Attention Score Table
        wei = q @ k.transpose(-2, -1) * q.shape[-1]**-0.5
        # Masked Attention
        wei = wei.masked_fill(self.tril[:n_context, :n_context] == 0, float('-inf'))
        # Aggregation
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out

""" multiple heads of self-attention in parallel"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, EMBEDDING_SIZE) # back to original size (see 3b1b Value↑ matrix)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        # x.shape => [BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE]
        self.out = torch.cat([h(x) for h in self.heads], dim=-1) # (BATCH_SIZE, CONTEXT_SIZE, num_heads*head_size)
        self.out = self.dropout(self.proj(self.out)) # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        return self.out

class FeedForward(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, in_feat * 4),
            nn.ReLU(),
            nn.Linear(4 * in_feat, in_feat),
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1E-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

# Transformer Block: Communication (MultiHead Attention) followed by computation (MLP - FeedForward)
class Block(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(EMBEDDING_SIZE)

        self.ln1 = LayerNorm(EMBEDDING_SIZE)
        self.ln2 = LayerNorm(EMBEDDING_SIZE)
    
    def forward(self, x):
        # x + because their are residual connections around Masked Multi-Head Attention and Feed Forward (see Transformer Architecture)
        x = x + self.sa_heads(self.ln1(x)) # (BATCH_SIZE, CONTEXT_SIZE, num_heads*head_size)
        x = x + self.ffwd(self.ln2(x)) # (BATCH_SIZE, CONTEXT_SIZE, num_heads*head_size)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # add an Embedding Table for Character Embedding
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(CONTEXT_SIZE, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[Block(NUM_HEADS, HEAD_SIZE) for _ in range(BLOCK_COUNT)])
        self.ln_f = nn.LayerNorm(EMBEDDING_SIZE) # final layer norm
        self.lm_head = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, y=None):
        n_batch, n_context = x.shape

        tok_emb = self.token_embedding_table(x) # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        pos_emb = self.position_embedding_table(torch.arange(0, n_context, device=device)) # position embedding for each char in CONTEXT (CONTEXT_SIZE, EMBEDDING_SIZE)
        x = tok_emb + pos_emb # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        x = self.blocks(x)
        x = self.ln_f(x) # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        logits = self.lm_head(x) # (BATCH_SIZE, CONTEXT_SIZE, VOCAB_SIZE)
        
        if y is None:
            loss = None
        else:
            logits = logits.view(n_batch*n_context, VOCAB_SIZE)
            y = y.view(n_batch*CONTEXT_SIZE)
            loss = F.cross_entropy(logits, y)

        return logits, loss
    
    def generate(self, previous_text, max_new_tokens):
        output = previous_text
        for _ in tqdm(range(max_new_tokens)):
            last_tokens = torch.tensor(encode(output[-CONTEXT_SIZE:]), device=device)
            
            # add batch dimension and feed to model
            logits, _ = self(last_tokens.view(1, -1))
            probs = F.softmax(logits, dim=-1)
            probs_next_char = probs[0, -1]
            new_char = itos[torch.multinomial(probs_next_char, num_samples=1).item()]

            output += new_char

        return output

model = Decoder()
model.to(device)

if not os.path.exists(os.path.abspath("./models/model(HarryPotter).pth")):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in tqdm(range(MAX_STEPS)):
        # calculate loss every once in a while
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"Step {step}/{MAX_STEPS}) train: {losses['train']:.4f}, val: {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "models/model(HarryPotter).pth")

else:
    model.load_state_dict(torch.load("./models/model(HarryPotter).pth", map_location=torch.device(device)))
    model.eval()

# Inference (Generate Harry Potter'ish text)
output = model.generate("\n", 2000)
outputFile = open("./output/HarryPotterText.txt", "a")
outputFile.write(output)
print(output)
outputFile.close()