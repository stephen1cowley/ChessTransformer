"""
File: trainer.py
Author: Stephen Cowley
Date: 20th Aug 2023
Description: The final chess engine model is trained using a list of games, and the model state dictionary saved to file.
"""

from final_model import *
import time

# Hyperparameters
max_iters               = 100
eval_interval           = 200 # interval to return estimate of val loss
eval_iters              = 100 # number of data batches to estimate val loss on
learning_rate           = 6e-4

# Create games list dataset
games_list = []
with open('data\\NicerData\\FormattedGamesList.csv', "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        games_list.append(line)

# Train and test splits
data = torch.tensor([encode(game) for game in games_list], dtype=torch.long)
n = int(0.9*len(data))
data = data[torch.randperm(data.size()[0])] # Shuffle data
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][:block_size] for i in ix])
    y = torch.stack([data[i][1:block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Context manager. Everything that happens in this function we will not do backprop on. (leading to much more efficient memory usage)
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = TransformerModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

time_buffer = time.time()
for iter in range(max_iters):

    # Every once in a  while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"Time passed: {np.around(((time.time() - time_buffer)/60), 1)} mins")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

device = 'cpu'
m = model.to(device)

# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.trainer_generate(context, max_new_tokens=58, device=device)[0].tolist()))

# Save the state dictionary to file
torch.save(model.state_dict(), "scaled_model_dict_2.pt")