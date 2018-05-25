import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pydot
import graphviz
from keras.utils import plot_model
import pandas

class KerasRNN:
	def __init__(self, dataframe, look_back):
		# fix random seed for reproducibility
		numpy.random.seed(7)
		
		self.dataset = dataframe.values
		self.dataset = self.dataset.astype('float')
		self.dataset = self.dataset[:]
		# normalize the dataset
		self.scaler_events = MinMaxScaler(feature_range=(0, 1))
		self.scaler_weekday = MinMaxScaler(feature_range=(0, 1))
		self.scaler_monthday = MinMaxScaler(feature_range=(0, 1))
		
		self.dataset[:,0] = self.scaler_events.fit_transform(self.dataset[:,0])
		self.dataset[:,1] = self.scaler_weekday.fit_transform(self.dataset[:,1])
		self.dataset[:,2] = self.scaler_monthday.fit_transform(self.dataset[:,2])
		
		# split into train and test sets
		train_size = int(len(self.dataset) * 0.75)
		test_size = len(self.dataset) - train_size
		train, test = self.dataset[0:train_size,:], self.dataset[train_size:len(self.dataset),:]

		# reshape into X=t and Y=t+1
		self.look_back = look_back
		self.trainX, self.trainY = self.create_dataset(train)
		self.testX, self.testY = self.create_dataset(test)
		# reshape input to be [samples, time steps, features]

		'''
		self.trainX = numpy.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
		self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
		'''

		self.model = Sequential()

		self.trainPredict = None
		self.testPredict = None

	def create_dataset(self, dataset):
		# convert an array of values into a dataset matrix
		dataX, dataY = [], []
		for i in range(len(dataset)-self.look_back):
			a = dataset[i:(i+self.look_back), 0:]
			dataX.append(a)
			dataY.append(dataset[i + self.look_back, 0])
		return numpy.array(dataX), numpy.array(dataY)

	def create_and_fit_LSTM(self, epochs):
		# create and fit the LSTM network
		self.model.add(LSTM(4, input_shape=(self.trainX.shape[1], self.trainX.shape[2])))
		self.model.add(Dense(1))
		self.model.compile(loss='mean_squared_error', optimizer='adam')
		self.model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=1, verbose=2)

	def make_predictions(self):
		# make predictions
		self.trainPredict = self.model.predict(self.trainX)
		self.testPredict = self.model.predict(self.testX)
		# invert predictions

		self.trainPredict = self.scaler_events.inverse_transform(self.trainPredict)
		self.trainY = self.scaler_events.inverse_transform(self.trainY)
		self.testPredict = self.scaler_events.inverse_transform(self.testPredict)
		self.testY = self.scaler_events.inverse_transform(self.testY)
		# calculate root mean squared error
		
		trainScore = math.sqrt(mean_squared_error(self.trainY, self.trainPredict))
		print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(self.testY, self.testPredict))
		print('Test Score: %.2f RMSE' % (testScore))

	def plot_graph(self):
		# shift train predictions for plotting
		trainPredictPlot = numpy.empty_like(self.dataset)
		trainPredictPlot[:, :] = numpy.nan
		trainPredictPlot[self.look_back:len(self.trainPredict)+self.look_back, :] = self.trainPredict

		print len(self.dataset)
		print len(self.trainPredict)
		print len(self.testPredict)
		# shift test predictions for plotting
		testPredictPlot = numpy.empty_like(self.dataset)
		testPredictPlot[:, :] = numpy.nan
		testPredictPlot[len(self.trainPredict)+(self.look_back*2):len(self.dataset), :] = self.testPredict
		# plot baseline and predictions
		plt.plot(self.scaler_events.inverse_transform(self.dataset)[:,0])
		plt.plot(trainPredictPlot)
		plt.plot(testPredictPlot)
		plt.show()

	def load_model(self, model_name):
		self.model = load_model(model_name+'.h5')

	def save_model(self, model_name):
		self.model.save(model_name+'.h5')

	def plot_model(self, model_name):
		plot_model(self.model, to_file=model_name+'.png')

if __name__ == '__main__':
	dataframe = pandas.read_csv("../Event Action Datasets From June 2016/MailRead_June2016.csv", usecols=[2,3,4], engine='python')
	rnn = KerasRNN(dataframe, 5)
	rnn.create_and_fit_LSTM(150)
	rnn.make_predictions()
	count_correct = 0

	for i in range(len(rnn.trainPredict)):
		if rnn.trainY[i]-rnn.trainPredict[i][0] < 150000:
			count_correct += 1
	for i in range(len(rnn.testPredict)):
		if rnn.testY[i]-rnn.testPredict[i][0] < 150000:
			count_correct += 1
	print "Accuracy: ", float(count_correct)*100/float(len(rnn.testPredict)+len(rnn.trainPredict))

	rnn.plot_graph()
	rnn.save_model("MailRead")