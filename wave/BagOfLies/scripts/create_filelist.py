import pandas as pd


if __name__ == '__main__':
	
	df = pd.read_csv('../data/annotations_fixed.csv')
	with open("../data/file_list.txt", "w+") as fl, open("../data/labels.txt", "w+") as labels, open("../data/splits.txt", "w+") as splits:
		for i in range(df.shape[0]):
			v = "/".join( df['video'][i].split("/")[3:5] )
			### file_list -> sample start end
			fl.write(v + " " + str(df['start'][i]) + " " + str(df['end'][i]) + "\n")
			labels.write(v + " " + str(df['truth'][i]) + "\n")
			splits.write(v + " " + str(df['splitB'][i]) + "\n")