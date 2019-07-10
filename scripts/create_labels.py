
import os
import numpy as np
import pandas as pd

## some games did not have standard naming
filename   = 'labels_canonical.csv'
labels_dir = os.path.join(os.getcwd(), 'labels/')

if not os.path.exists(labels_dir):
	os.makedirs(labels_dir)


df = pd.read_csv(filename)
print("Shape of dataframe : {}".format(df.shape))

for _, row in df.iterrows():
	game_player = row['gamename'][3:].upper() + '-' + row['gamename'][:3] + '_' + str(row['player'])
	role   	    = str(row['role_spy']) 
	#print(game_player, role)
	
	with open(labels_dir + game_player + '.txt', 'w') as fname:
		fname.write(role)