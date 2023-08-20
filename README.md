# ChessTransformer
This is a transformer model based on the 2017 paper 'Attention Is All You Need' by Vaswani et. al. The architecture used here is formed of 3 decoder blocks consisting of: 16-head multi-head attention with head size 64, followed by a linear layer to form logits of size 4375 (the number of unique PGN chess moves supported by the model). The block size (maximum context length of half-moves to predict the target) is 58.

Much of the model is inspired by Andrej Karpathy's video, 'Let's build GPT: from scratch, in code, spelled out.' In essence, I build a next-word-predictor similar to GPT but 'words' are chess half-moves. Although this isn't a standard way of creating a chess engine (compared with techniques such as Q-learning etc.), the power of multi-head attention in this context can still be displayed.

# Data
All data is taken from pgnmentor.com. Around 340,000 1. d4, master-level tournament games of length 60 or more half moves were taken and converted into csv format. Only a portion of these games are used, depending on training time.

# Training
When trained with a batch size of 32, over 6000 iterations (around 40 minutes on an 8GB GPU), the final validation loss (cross-entropy) was 2.4. Note that if the block size (maximum context length) was reduced to 28, and batch size 16, the loss reached around 1.8 over just 2000 iterations.

# Other Notes
The model is not fed the rules of chess. Therefore the model may sometimes produce an illegal move. However, in chess_engine.py, where one can play against the trained engine, the generator has an added move-legality check to make the game playable. 