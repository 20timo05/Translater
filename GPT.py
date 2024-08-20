import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from parameters import (
    EMBEDDING_SIZE,
    DROPOUT,
    CONTEXT_SIZE,
    device,
    VOCAB_SIZE,
    NUM_HEADS,
    HEAD_SIZE,
    ENCODER_BLOCK_COUNT,
    DECODER_BLOCK_COUNT,
)

""" Multiple Heads of Self-Attention in parallel """
class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, masking=False):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_size = head_size
        self.masking = masking

        # query, key & value matrix for n heads that can be processed in parallel
        self.query = nn.Linear(EMBEDDING_SIZE, num_heads*head_size, bias=False)
        self.key = nn.Linear(EMBEDDING_SIZE, num_heads*head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, num_heads*head_size, bias=False)
        
        self.dropout1 = nn.Dropout(DROPOUT)

        # Only For Multi Head
        self.proj = nn.Linear(num_heads*head_size, EMBEDDING_SIZE) # back to original size (see 3b1b Value↑ matrix)
        self.dropout2 = nn.Dropout(DROPOUT)

        if self.masking:
            # since it's not a parameter of the model => register as buffer
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)).view(
                    1, 1, CONTEXT_SIZE, CONTEXT_SIZE
                ),
            )
    
    def forward(self, x, cross_x=None):
        n_batch, n_context, n_emb = x.shape

        q = self.query(x).view(n_batch, n_context, self.num_heads, self.head_size).transpose(1, 2)

        if cross_x is None:
            # Self-Attention
            k = self.key(x).view(n_batch, n_context, self.num_heads, self.head_size).transpose(1, 2)
            v = self.value(x).view(n_batch, n_context, self.num_heads, self.head_size).transpose(1, 2)
        else:
            # Cross-Attention
            n_context_cross = cross_x.size(1)
            k = self.key(cross_x).view(n_batch, n_context_cross, self.num_heads, self.head_size).transpose(1, 2)
            v = self.value(cross_x).view(n_batch, n_context_cross, self.num_heads, self.head_size).transpose(1, 2)

        # q, k, v.shape = [n_batch, num_heads, n_context, head_size]

        # Attention Score Table (Scaled dot-product attention   )
        wei = q @ k.transpose(-2, -1) * q.shape[-1]**-0.5 # [n_batch, num_heads, n_context, n_context or n_context_cross]
        
        if self.masking and cross_x is None: # Masked Attention - apply only for Self-Attention
            wei = wei.masked_fill(self.tril[:, :, :n_context, :n_context] == 0, float('-inf'))
        
        # Aggregation
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout1(wei)

        out = wei @ v # [n_batch, num_heads, n_context, head_size]
        out = out.transpose(1, 2).reshape(n_batch, n_context, self.num_heads*self.head_size)

        out = self.dropout2(self.proj(out)) # (n_batch, n_context, EMBEDDING_SIZE)
        return out


class FeedForward(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, in_feat * 4),
            nn.ReLU(),
            nn.Linear(4 * in_feat, in_feat),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.masked_sa_heads = CausalSelfAttention(n_heads, head_size, masking=True)
        self.ln1 = nn.LayerNorm(EMBEDDING_SIZE)

        self.sa_heads = CausalSelfAttention(n_heads, head_size, masking=False)
        self.ln2 = nn.LayerNorm(EMBEDDING_SIZE)

        self.ffwd = FeedForward(EMBEDDING_SIZE)
        self.ln3 = nn.LayerNorm(EMBEDDING_SIZE)
    
    def forward(self, x, enc_output):
        # x + because there are residual connections around Multi-Head Attention and Feed Forward (see Transformer Architecture)
        x = x + self.masked_sa_heads(self.ln1(x))
        x = x + self.sa_heads(self.ln2(x), enc_output)
        x = x + self.ffwd(self.ln3(x)) # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DecoderBlock(NUM_HEADS, HEAD_SIZE) for _ in range(DECODER_BLOCK_COUNT)]
        )

    def forward(self, dec_input, enc_output):
        x = dec_input
        for block in self.blocks:
            x = block(x, enc_output) # [BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE]
            
        return x

""" Transformer Block: Communication (MultiHead Attention) followed by computation (MLP - FeedForward) """
class EncoderBlock(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.sa_heads = CausalSelfAttention(n_heads, head_size, masking=False)
        self.ffwd = FeedForward(EMBEDDING_SIZE)

        self.ln1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.ln2 = nn.LayerNorm(EMBEDDING_SIZE)
    
    def forward(self, x):
        # x + because their are residual connections around Multi-Head Attention and Feed Forward (see Transformer Architecture)
        x = x + self.sa_heads(self.ln1(x)) # (BATCH_SIZE, CONTEXT_SIZE, num_heads*head_size)
        x = x + self.ffwd(self.ln2(x)) # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        return x

""" Encoder of Transformer necessary for Cross Attention """
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(NUM_HEADS, HEAD_SIZE) for _ in range(ENCODER_BLOCK_COUNT)]
        )
    
    def forward(self, x):
        x = self.blocks(x) # [BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE]
        return x



""" Combines Encoder and Decoder into full transformer model """
class GPT(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        # add an Embedding Table for Character Embedding
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(CONTEXT_SIZE, EMBEDDING_SIZE)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.lm_head = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE, bias=False)
        # weight sharing (use same weights for Input Embeddings (token_embedding_table) and lm_head)
        self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, enc_input, dec_input, targets=None, ignore_index=None):
        n_batch, enc_n_context = enc_input.shape
        n_batch, dec_n_context = dec_input.shape

        """ Embedding input sequences & add positional encoding """
        enc_emb = self.token_embedding_table(enc_input)  # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        enc_pos_emb = self.position_embedding_table(
            torch.arange(0, enc_n_context, device=device)
        )  # position embedding for each char in CONTEXT (CONTEXT_SIZE, EMBEDDING_SIZE)
        enc_input = enc_emb + enc_pos_emb  # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)

        dec_emb = self.token_embedding_table(dec_input)  # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        dec_pos_emb = self.position_embedding_table(
            torch.arange(0, dec_n_context, device=device)
        )  # position embedding for each char in CONTEXT (CONTEXT_SIZE, EMBEDDING_SIZE)
        dec_input = dec_emb + dec_pos_emb  # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)

        """ Calculate Encoder ouput (needs to be passed do decoder for Cross Attention) """
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output)

        logits = self.lm_head(dec_output) # (BATCH_SIZE, CONTEXT_SIZE, VOCAB_SIZE)
        if targets is None:
            loss = None
        else:
            logits = logits.view(n_batch * dec_n_context, VOCAB_SIZE)
            targets = targets.view(n_batch * dec_n_context)
            loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
        
        return logits, loss
    
    def translate(self, english_text, max_new_tokens=100):
        self.eval()

        eng_enc = self.tokenizer.encode(english_text)
        assert len(eng_enc) <= CONTEXT_SIZE, f"Sentence is too long! (max_tokens={CONTEXT_SIZE})"
        eng_enc = torch.tensor(eng_enc, device=device).view(1, -1)

        start_token = [
            key for key, value in self.tokenizer.vocab.items() if value == b"<|STARTOFTEXT|>"
        ][0]
        end_token = [
            key for key, value in self.tokenizer.vocab.items() if value == b"<|ENDOFTEXT|>"
        ][0]

        output_tokens = torch.tensor([start_token])
        for _ in range(max_new_tokens):
            last_tokens = output_tokens[-CONTEXT_SIZE:].view(1, -1)
            logits, _ = self(eng_enc, last_tokens)
            logits = logits[:, -1, :] # only use prediction using all tokens in context and predict actual next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)[0]

            if next_token[0].item() == end_token:
                break

            output_tokens = torch.cat((output_tokens, next_token))
        
        translation = self.tokenizer.decode(output_tokens.tolist()[1:])
        
        self.train()
        
        return translation