import os
import numpy


if __name__ == '__main__':
	
	with open("../data/README.txt", 'r') as f:
		lines = list(filter(lambda x: x[0]!='+', f.readlines()))
		lines = list(map(lambda x: x.split("|")[1:3], lines))
	
	for line in lines:
		line[0] = line[0].strip()
		line[1] = line[1].strip()
	
	with open("R.txt", 'w') as f:
		for line in lines:
			f.write('-'.join(line)+'\n')