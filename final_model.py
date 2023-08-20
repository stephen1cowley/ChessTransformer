"""
File: final_model.py
Author: Stephen Cowley
Date: 20th Aug 2023
Description: The final transformer model that takes a sequence of chess moves and outputs the next move.
Inspired by Andrej Karpathy's video, 'Let's build GPT: from scratch, in code, spelled out.'
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import csv

# Hyperparameters
batch_size              = 32 # number of independent sequences run in parallel
block_size              = 58 # maximum context length for predicitons
n_embd                  = 1024 # embedding dimension
n_head                  = 16
p_dropout               = 0.2 # probability of setting a weight to zero during training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1459)

# Open list of unique moves
unique_moves = []
with open('data\\NicerData\\UniqueMoves.csv', "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        unique_moves.append(line)
unique_moves = unique_moves[0]
vocab_size = len(unique_moves)

# Encoding and decoding from move names to unique integers
stoi = { m:i for i,m in enumerate(unique_moves)}
itos = { i:m for i,m in enumerate(unique_moves)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([(itos[i]+', ') for i in l])

class Head(nn.Module):
    # One head of self attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x):
        B,T,C  = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 #Produces a (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v #Produces a (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    # Multiple heads of self attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    # A simple linear layer followed by non-linearity

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(p_dropout), # Right before the connection into the residual pathway
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer block - communication followed by computation

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head),
            Block(n_embd, n_head),
            Block(n_embd, n_head),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) #A linear layer

    def forward(self, idx, targets=None, device=device):
        # both idx and targets are (B,T) tensors of integers
        # set default of targets as None
        B, T = idx.shape
        # both idx and targets are (B,T) tensors of integers
        # set default of targets as None
        tok_emb = self.token_embedding_table(idx) # (B,T,C)  batch, time, channels (vocab size) which is 65
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
 
            # loss = F.cross_entropy(logits, targets) # logits are predictions, targets are the targets. Thus find the LOSS. This is the quality of the logits wrt the targets
            # Now can go to the pytorch loss entropy documentation.
            # But needded input in form BCT

            B, T, C = logits.shape
            logits = logits.view(B*T, C) #This is now just a 2D array, sort of stretching the BXT array into a single list but preserving the C as the second dim.
            #Now will work for pytorch
            targets = targets.view(B*T) #Similarly 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def trainer_generate(self, idx, max_new_tokens, device):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx, device=device) #idx is indices. self(idx) goes to the forward function
            # Focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
        
            # Sample from the distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    def engine_generate(self, idx, max_new_tokens, board, device):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx, device=device) # idx is indices. self(idx) goes to the forward function
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            
            # Find legal moves based on the current position
            legal_moves = list(board.legal_moves)
            legal_list = [board.san(move) for move in legal_moves]
            coded_legal_list = encode(legal_list)
            
            # Sets probabilities of illegal moves to be zero
            for i, _ in enumerate(probs.tolist()[0]):
                if i not in coded_legal_list:
                    probs[0,i] = 0

            # Sample the best move
            idx_next = torch.argmax(probs).view(1,1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx