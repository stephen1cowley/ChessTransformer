"""
File: data_handler_2.py
Author: Stephen Cowley
Date: 20th Aug 2023
Description: The large CSV file is formatted to include only games of a certain length, and a list of all unique moves is made.
"""

import csv

games_file = 'data\\NicerData\\LargeDataset.csv'

# Create list of games from the csv
games_list = []
with open(games_file, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        if line != []:
            games_list.append(line)

# Make all games length L
L = 60
new_games_list = []
for game in games_list:
    if len(game) >= L:
        new_games_list.append(game[:L])

# Find list of unique moves
unique_moves = []
for element in new_games_list:
    for sub_element in element:
        if sub_element not in unique_moves:
            unique_moves.append(sub_element)
unique_moves = [unique_moves]

# Write the two lists to csv
with open('FormattedGamesList.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(new_games_list)

with open('UniqueMoves.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(unique_moves)