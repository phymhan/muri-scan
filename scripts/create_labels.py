
import os
import numpy as np
import pandas as pd

## some games did not have standard naming
filename   = 'labels_canonical.csv'

df = pd.read_csv(filename)
print("Shape of dataframe : {}".format(df.shape))

with open('labels.txt', 'w') as fname:
	for _, row in df.iterrows():
		game_player = row['gamename'][3:].upper() + '-' + row['gamename'][:3] + '_' + str(row['player'])
		role   	    = str(row['role_spy']) 
		#print(game_player + ' ' + role)
		fname.write(game_player + ' ' + role + '\n')