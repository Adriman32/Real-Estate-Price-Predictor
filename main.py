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

def createLayers(hidden_layer_nodes, max_layers):
	counter = 0
	arr_hoder = []

	# Creates permutations of all nodes	and layers
	for node in hidden_layer_nodes:
		arr_hoder.append(node) 
	if(max_layers > 1):
		for x in range(0,max_layers-1):
			for i in range(counter,len(arr_hoder)):
				if(i == counter):
					counter = len(arr_hoder)
				for node in hidden_layer_nodes:
					temp_arr = []
					temp_arr.append(arr_hoder[i])
					temp_arr.append(node)
					arr_hoder.append(temp_arr)

	# Converts from List of Lists to 1D lists
	layer_arr = []
	for layer in arr_hoder:
		if(type(layer) == np.int32):
			layer_arr.append(layer)
		else:
			ender = False
			counter = 0
			checker = layer.copy()
			temp_arr = []
			while(not(ender)):
				if(type(checker) == np.int32):
					temp_arr.append(checker)
					ender = True
				else:
					flag = 0
					counter = len(checker)
					loc = 0
					while(loc < counter):
						if(type(checker[loc])==np.int32):
							temp_arr.append(int(checker[loc]))
							loc+=1
						else:
							holder = checker.copy()
							checker = checker[loc]
							flag = loc+1
							loc = 0
					if(flag > 0):
						checker = holder[flag].copy()
					else:
						ender = True
			layer_arr.append(temp_arr)		
	return layer_arr


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

	
	#hidden_layer_nodes = np.array([1,2,4,8,16,32,64,128,256,512,1024])
	max_hidden_layers = 3
	hidden_layer_nodes = np.array([256,512,1024])
	
	model_arr = []
	layer_arr = createLayers(hidden_layer_nodes, max_hidden_layers)
	for layer in layer_arr:
		model_arr.append(annTrainer(x_train,y_train,layer))

	pred_arr = []
	for model in model_arr:
		pred_arr.append(model.predict(x_test))


	mse_arr = []
	diff_arr = []
	for i in range(0,len(pred_arr)):
		mse_arr.append(model_arr[i].loss_)

	for i in range(0,len(y_test)):
		diff_arr.append(np.mean(abs(y_test[i]-pred_arr[i])))

	min_mse_loc = np.argmin(mse_arr)
	min_diff_loc = np.argmin(diff_arr)

	print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
	print('Best Model (MSE):',model_arr[min_mse_loc])
	print('MSE:', mse_arr[min_mse_loc])
	print('Best Model (Diff):',model_arr[min_diff_loc])
	print('Diff:', diff_arr[min_diff_loc])

	plt.figure(1)
	plt.plot(range(0,model_arr[min_mse_loc].n_iter_),model_arr[min_mse_loc].loss_curve_,label='MSE Loss')
	plt.title('MSE Loss over 200 Iterations')
	plt.xlabel('Epochs')
	plt.ylabel('MSE')
	plt.legend()

	plt.figure(2)
	plt.plot(range(0,len(diff_arr)),diff_arr,label = 'Avg Diff Per Model')
	plt.plot(range(0,len(mse_arr)),mse_arr,label = 'MSE Per Model')
	plt.title('Model Loss')
	plt.xlabel('Model Number')
	plt.ylabel('Average Difference')
	plt.legend()
	plt.show()

		



	'''
	print("Predicted:", ann_model.predict(x_test[5].reshape(1,-1)),"Actual:",y_test[5])


	plt.scatter(y_test,pred,label="Predictions")
	plt.plot(y_test,y_test,'r',label="Actual")
	plt.title("House Price Per Unit Area")
	plt.xlabel("Actual Price Per Unit Area")
	plt.ylabel("Predicted Price Per Unit Area")
	plt.legend()
	plt.show()
	
	'''
	print("Elapsed Time:",time.perf_counter()-start_time,"seconds.")

if __name__ == '__main__':
	main()