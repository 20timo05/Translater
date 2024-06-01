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


""" Multiple Heads of Attention that are processed in parallel """
class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, masking=False):
        super().__init__()

        self.masking = masking

        # Single Heads in parallel
        self.query = torch.randn([num_heads, EMBEDDING_SIZE, head_size]) * 0.02
        self.key = torch.randn([num_heads, EMBEDDING_SIZE, head_size]) * 0.02
        self.value = torch.randn([num_heads, EMBEDDING_SIZE, head_size]) * 0.02

        self.dropout1 = nn.Dropout(DROPOUT)
        if self.masking:
            self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
        
        # Only For Multi Head
        self.proj = nn.Linear(num_heads*head_size, EMBEDDING_SIZE) # back to original size (see 3b1b Value↑ matrix)
        self.dropout2 = nn.Dropout(DROPOUT)
    
    def forward(self, x, cross_x=None):
        n_batch, n_context, n_emb = x.shape
        num_heads, head_size = self.query.shape[0], self.query.shape[-1]

        # (num_heads, n_batch, n_context, head_size)
        q = torch.einsum('bxy,iyk->bxik', (x, self.query)).view(num_heads, n_batch, n_context, head_size)
        if cross_x is None:
            # Self-attention
            k = torch.einsum('bxy,iyk->bxik', (x, self.key)).view(num_heads, n_batch, n_context, head_size)
            v = torch.einsum('bxy,iyk->bxik', (x, self.value)).view(num_heads, n_batch, n_context, head_size)
        else:
            # Cross-attention (key and value are from cross_x)
            k = torch.einsum('bxy,iyk->bxik', (cross_x, self.key)).view(num_heads, n_batch, cross_x.size(1), head_size)
            v = torch.einsum('bxy,iyk->bxik', (cross_x, self.value)).view(num_heads, n_batch, cross_x.size(1), head_size)

        
        wei = q @ k.transpose(-2, -1) * q.shape[-1]**-0.5 # (num_heads, n_batch, n_context, n_context)
    
        if self.masking and cross_x is None: # Apply only for Self-Attention
            wei = wei.masked_fill(self.tril[:n_context, :n_context] == 0, float('-inf'))
    
        wei = F.softmax(wei, dim=-1) # (num_heads, n_batch, n_context, n_context)
        wei = self.dropout1(wei)

        self.out = wei @ v # (num_heads, n_batch, n_context, head_size)
        self.out = self.out.view(n_batch, n_context, num_heads*head_size)
        self.out = self.dropout2(self.proj(self.out)) # (n_batch, n_context, EMBEDDING_SIZE)
        return self.out


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
        self.blocks = nn.Sequential(
            *[DecoderBlock(NUM_HEADS, HEAD_SIZE) for _ in range(DECODER_BLOCK_COUNT)]
        )

    def forward(self, dec_input, enc_output):
        x = self.blocks(dec_input, enc_output) # [BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE]
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
    def __init__(self):
        super().__init__()

        # add an Embedding Table for Character Embedding
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(CONTEXT_SIZE, EMBEDDING_SIZE)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.lm_head = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)

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
    
    def forward(self, enc_input, dec_input, targets=None):
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
            y = y.view(n_batch * CONTEXT_SIZE)
            loss = F.cross_entropy(logits, y)
        
        return logits, loss