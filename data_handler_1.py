"""
File: data_handler_1.py
Author: Stephen Cowley
Date: 20th Aug 2023
Description: The raw PGN files are converted into a digestable CSV file.
"""

import chess.pgn
import csv
import os
import time
import numpy as np

listed_files = []
game_list = []
k = 0
time_buffer = time.time()

# Create list of files to process
for subdir, dirs, files in os.walk('data\\MasterGames'):
    for file in files:
        filepath = subdir + os.sep + file
        listed_files.append(filepath)

# Manually convert PGN to CSV
for file in listed_files:
    pgn = open(file)
    
    while(True):
        if (k % 200 == 100):
            # Monitor progress of algorithm
            print(f"Games processed = {k}")
            gps = np.around((float(k) / (time.time() - time_buffer)), 1)
            print(f"Games processed per second = {gps}", "\n")
        k += 1
        
        try:
            # Read a single game
            cur_game = chess.pgn.read_game(pgn)
            pgn_moves = str(cur_game.mainline_moves())
        except:
            # No more games left to read in this file; switch to next file
            break
        
        move_list = []
        new_move = False
        move_to_append = ''

        for char in pgn_moves:
            # Iterate through all characters in the PGN game
            if char == ' ':
                new_move = True
                if move_to_append != '':
                    if move_to_append[-1] != '.':
                        move_list.append(move_to_append)
                    move_to_append = ''
            if new_move == True:
                if char != ' ':
                    move_to_append += char

        game_list.append(move_list)

with open('LargeDataset.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(game_list)