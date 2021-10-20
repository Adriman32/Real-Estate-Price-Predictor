import pickle
import sklearn.neural_network as nnet
import sklearn.preprocessing as preproc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys


def dataFrameFromFile(filename,verbose=True):
	df = pd.read_csv(filename)
	if(verbose):
		print('Loaded File',filename,"of size", df.shape)
	return df

def splitDF(df,test_size=0.3,verbose=False):
	numTotal = len(df)
	numSplit = int(numTotal*test_size)
	full_test_df = pd.DataFrame()

	for i in range(0,numSplit):
		randNum = random.randint(0,numTotal-i-1)
		if(verbose):
			print ("Selected Record:",randNum)
		selected_df = df.iloc[randNum]
		full_test_df = full_test_df.append(selected_df)

		df = df.drop(randNum)
		df = df.reset_index(drop=True)

	full_test_df.reset_index(inplace=True,drop=True)

	x_test = full_test_df.drop(columns="Price Per Unit Area")
	y_test = full_test_df["Price Per Unit Area"]

	x_train = df.drop(columns="Price Per Unit Area")
	y_train = df["Price Per Unit Area"]

	x_train = preproc.MinMaxScaler((-1,1)).fit_transform(x_train)
	y_train = y_train.to_numpy()
	x_test = preproc.MinMaxScaler((-1,1)).fit_transform(x_test)
	y_test = y_test.to_numpy()

	return (x_train,y_train,x_test,y_test)


def main():

	filename = 'models/MLP_[128, 512, 512]_MSE_1.2878603253710939'

	pre_trainied = pickle.load(open(filename,'rb'))
	filename = "data.csv"
	full_df = dataFrameFromFile(filename)
	full_df.drop(columns=['No'],inplace=True)
	full_df.rename(columns={
					"X1 transaction date":"Transaction Date",
					"X2 house age":"House Age",
					"X3 distance to the nearest MRT station":"Distance to MRT",
					"X4 number of convenience stores":"Num Convenience Stores",
					"X5 latitude":"Latitude",
					"X6 longitude":"Longitude",
					"Y house price of unit area":"Price Per Unit Area"
					},inplace=True)

	x_train,y_train,x_test,y_test = splitDF(full_df,test_size=0.75)

	
	num_predictions = int(len(x_test)*0.75)
	val_arr = []
	pred_arr = []
	for i in range(0,num_predictions):
		val = random.randint(0,len(x_test)-1)
		predVal = int(pre_trainied.predict(x_test[val].reshape(1,-1)))
		testVal = y_test[val]
		print("Predicted:",predVal ,"Actual:",testVal)
		val_arr.append(testVal)
		pred_arr.append(predVal)

	plt.scatter(val_arr,pred_arr,label='Predictions')
	plt.plot(y_test,y_test,label='Actual Values',c='RED')
	plt.xlabel('Actual Values')
	plt.ylabel('Predicted Values')
	plt.legend()
	plt.title('Predicted vs Actual Real Estate Prices')
	plt.show()
	
if __name__=='__main__':
	main()