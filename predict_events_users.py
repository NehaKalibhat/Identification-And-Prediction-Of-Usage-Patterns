from KerasRNN import KerasRNN as rnn_days
from KerasRNN_Hour_Week_Day import KerasRNN as rnn_hours
from KerasRNN_users import KerasRNN as rnn_users
import pandas
import numpy
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from ttkcalendar import get_selection

def predict_unknown(rnn, startX, count):
	unknownX = numpy.array([],dtype='float')
	unknownY = numpy.array([],dtype='float')
	while count:
		if len(unknownX) == 0:
			unknownX = startX
			unknownX = numpy.reshape(unknownX , (1,1)+unknownX.shape)
		else:
			new_x = numpy.append(unknownX[-1][0][1:] , unknownY[-1])
			unknownX = numpy.append(unknownX, numpy.reshape(new_x , (1,1)+new_x.shape), axis = 0) 
		unknownY = numpy.append(unknownY, rnn.model.predict(numpy.reshape(unknownX[-1], (1,)+unknownX[-1].shape)))
		
		count -= 1 

	unknownY = rnn.scaler.inverse_transform(unknownY)
	unknownY = numpy.reshape(unknownY , unknownY.shape + (1,))

	return unknownY

def predict_unknown_rnn2(rnn, startX, last_date, count):
	unknownX = numpy.array([],dtype='float')
	unknownY = numpy.array([],dtype='float')
	weekdays = numpy.array(range(7)).astype('float')
	weekdays = rnn.scaler_weekday.fit_transform(weekdays)
	monthdays = numpy.array(range(1,32)).astype('float')
	monthdays = rnn.scaler_monthday.fit_transform(monthdays)
	hours = numpy.array(range(24)).astype('float')
	hours = rnn.scaler_weekday.fit_transform(hours)
					
	while count:
		weekday = weekdays[last_date.weekday()]
		monthday = monthdays[last_date.day-1]
		hour = hours[last_date.hour]

		if len(unknownX) == 0:
			new = numpy.array([weekday, monthday, hour, startX])
			unknownX = numpy.append(unknownX, new)
			unknownX = numpy.reshape(unknownX , (1,1)+unknownX.shape)
			
		else:
			new = numpy.array([weekday, monthday, hour, unknownY[-1]])
			unknownX = numpy.append(unknownX, numpy.reshape(new , (1,1)+new.shape), axis = 0) 
		
		unknownY = numpy.append(unknownY, rnn.model.predict(numpy.reshape(unknownX[-1], (1,)+unknownX[-1].shape)))
		
		last_date += datetime.timedelta(hours=1)
		count -= 1 

	unknownY = rnn.scaler_events.inverse_transform(unknownY)
	unknownY = numpy.reshape(unknownY , unknownY.shape + (1,))

	return unknownY

def plot_graph(rnn, unknownY=None):
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(rnn.dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[rnn.look_back:len(rnn.trainPredict)+rnn.look_back, :] = rnn.trainPredict

	print len(rnn.dataset)
	print len(rnn.trainPredict)
	print len(rnn.testPredict)
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(rnn.dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(rnn.trainPredict)+(rnn.look_back*2):len(rnn.dataset), :] = rnn.testPredict

	#shift unknown predictions for plotting 
	unknownPredictPlot = None
	if type(unknownY) != type(None):
		unknownPredictPlot = numpy.empty_like(numpy.append(numpy.reshape(rnn.dataset[:,-1], rnn.dataset[:,-1].shape+(1,))  , unknownY , axis = 0))
		unknownPredictPlot[:, :] = numpy.nan
		unknownPredictPlot[len(rnn.dataset):len(rnn.dataset)+len(unknownY), :] = unknownY
		plt.plot(unknownPredictPlot, label="Predictions on Unknown Data")

	# plot baseline and predictions
	try:
		plt.plot(rnn.scaler.inverse_transform(rnn.dataset)[:,-1],label="Actual Data")
	except:
		plt.plot(rnn.scaler_events.inverse_transform(rnn.dataset)[:,-1], label="Actual Data") 
		plt.plot(trainPredictPlot[:,3], label="Predictions on Training Dataset")
		plt.plot(testPredictPlot[:,3], label="Predictions on Testing Dataset")
	else:
		plt.plot(trainPredictPlot, label="Predictions on Training Dataset")
		plt.plot(testPredictPlot, label="Predictions on Testing Dataset")
	plt.legend(loc='best')
	plt.show()

dataframe1 = pandas.read_csv("../Event Action Datasets From June 2016/MailRead_June2016.csv", usecols=[2])
dataframe2 = pandas.read_csv("../Event Action Datasets From June 2016/MailRead_hours_March2017.csv", usecols=[2,3,4,5])
dataframe3 = pandas.read_csv("../Event Action Datasets From June 2016/MailRead_users_August2016.csv", usecols=[2])

rnn1 = rnn_days(dataframe1, 12)
rnn2 = rnn_hours(dataframe2, 1)
rnn3 = rnn_users(dataframe3, 12)
rnn1.load_model("MailRead_day")
rnn2.load_model("MailRead_hour_week_day")
rnn3.load_model("MailRead_users")

selected_date = get_selection()

last_date1 = datetime.datetime(year = 2017, month = 3, day = 20, hour = 0)
last_date2 = datetime.datetime(year = 2017, month = 4, day = 1, hour = 18)
last_date3 = datetime.datetime(year = 2017, month = 3, day = 31, hour = 0)

days1 = (selected_date - last_date1).days
hours2 = (selected_date - last_date2).total_seconds()//3600
days3 = (selected_date - last_date3).days

unknownY1 = predict_unknown(rnn1, rnn1.testY[-12:], days1)
rnn1.make_predictions()
plot_graph(rnn1, unknownY1)

unknownY2 = predict_unknown_rnn2(rnn2, rnn2.testY[-1], last_date2,hours2)
rnn2.make_predictions()
plot_graph(rnn2, unknownY2)

unknownY3 = predict_unknown(rnn3, rnn3.testY[-12:], days3)
rnn3.make_predictions()
plot_graph(rnn3, unknownY3) 

print "# of events on ",selected_date," : ",unknownY1[-1]
print "# of events on ",selected_date," : ",unknownY2[-1]
print "# of users on ",selected_date," : ",unknownY3[-1]
