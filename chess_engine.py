"""
File: chess_engine.py
Author: Stephen Cowley
Date: 20th Aug 2023
Description: A trained model plays a game against the user.
"""

import chess
from final_model import *

# Switch to cpu so that TransformerModel() and chess.Board() are on the same device
device = 'cpu'

model = TransformerModel()
model.load_state_dict(torch.load('trained_models\\scaled_model_dict.pt'))
model.eval()

# Initial move is 1. d4 made by the engine by default.
# This is because the model is only trained on 1. d4 games.
the_game = encode(['d4'])

# Begin the game
board = chess.Board()

print('\n')
print('Game has begun. The engine is white.', '\n')
print('1. d4')
board.push_san('d4')
print(board)

# No more than 57 half-moves will be allowed
for move_num in range(28):
    print('\n')
    while(True):
        try:
            # User inputs a move
            next_move = stoi[str(input("Enter your move: "))]
            the_game.append(next_move)
            board.push_san(itos[the_game[-1]])
            print('\n')
            print(f'{move_num+1}. {itos[the_game[-2]]} {itos[the_game[-1]]}')
            break
        except:
            # Either the move entered is illegal, or the entered move isn't in the engine's dictionary, or another error has occured.
            print('Error. Unrecognised move.\n')
            continue
    
    print(board)

    # Engine decides on a move based on the current position
    the_game.append(model.engine_generate(torch.tensor([the_game]), max_new_tokens=1, board=board, device=device)[0].tolist()[-1])

    print('\n')
    print(f'{move_num+2}. {itos[the_game[-1]]}')
    board.push_san(itos[the_game[-1]])
    print(board)