import numpy as np
import pandas as pd
import sklearn.neural_network as nnet
import sklearn.preprocessing as preproc
import matplotlib.pyplot as plt
import time
import random


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


def annTrainer(x_train,y_train,hidden_layer_nodes,verbose=False):
	ann_model = nnet.MLPRegressor(hidden_layer_sizes=(hidden_layer_nodes),activation='relu',max_iter=200,random_state=42)
	ann_model = ann_model.fit(x_train,y_train.ravel())
	#print(ann_model)
	if(verbose):
		print("\nArtificial Neural Network Statistics")
		print("Hidden Layer Nodes:",hidden_layer_nodes)
		print("MSE Loss:",ann_model.loss_)
	return (ann_model)



def main():
	start_time = time.perf_counter()
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

	x_train,y_train,x_test,y_test = splitDF(full_df)

	

	num_hidden_layers = np.array([1,2,3])
	#hidden_layer_nodes = [0,1,2,4,8,16,32,64,128,256,512,1024]
	hidden_layer_nodes = np.array([1,2,4])


	for layer in num_hidden_layers:
		for node in hidden_layer_nodes:
			temp_array=[]
			for i in range(0,layer):
				temp_array.append(node)
			print(temp_array)

	


	'''
	for layer in num_hidden_layers:
		for node in hidden_layer_nodes:
			temp_array=[]
			for i in range(0,layer):
				temp_array.append(node)
			print(temp_array)
	'''
	'''
	for i in num_hidden_layers:
		total_hidden_layers = []
		print("Reset")
		for layers in range(0,i):
			temp_layer = []
			for nodes in hidden_layer_nodes:
				temp_layer.append(nodes)
			total_hidden_layers.append(temp_layer)
		print(total_hidden_layers)
	#print(total_hidden_layers)

	'''
	hidden_layer_nodes = [200,200,200]
	ann_model = annTrainer(x_train,y_train,hidden_layer_nodes,verbose=True)
	
	pred = ann_model.predict(x_test)

	temp=[]

	for i in range(1,len(pred)):
		temp.append(y_test[i]-pred[i])

	print("\n\nMean Difference", np.mean(temp))
	print("Max Difference", max(temp),"\n\n")

	print("Predicted:", ann_model.predict(x_test[5].reshape(1,-1)),"Actual:",y_test[5])


	plt.scatter(y_test,pred,label="Predictions")
	plt.plot(y_test,y_test,'r',label="Actual")
	plt.title("House Price Per Unit Area")
	plt.xlabel("Actual Price Per Unit Area")
	plt.ylabel("Predicted Price Per Unit Area")
	plt.legend()
	plt.show()

	'''
	print("XTRAIN\n",x_train)
	print("YTRAIN\n",y_train)	
	print("XTEST\n",x_test)
	print("YTEST\n",y_test)	
	'''




	




	print("Elapsed Time:",time.perf_counter()-start_time,"seconds.")

if __name__ == '__main__':
	main()