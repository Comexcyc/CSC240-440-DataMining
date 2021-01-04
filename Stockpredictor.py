from datetime import date, timedelta
import datetime
import math
import numpy
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
print("please close all the graph for the code to continue running")
lt.style.use('fivethirtyeight')

today=date.today()

df = web.DataReader('AMZN', data_source='yahoo', start='2008-01-01', end=today)


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

data = df.filter(['Close'])

dataset=data.values

training_data_len = math.ceil(len(dataset)*.8)


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len,:]
x_train = []
y_train = []



for i in range(90,len(train_data)):
    x_train.append(train_data[i-90:i,0])
    y_train.append(train_data[i,0])


x_train, y_train = np.array(x_train), np.array(y_train)



x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

#MODEL construction
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train, batch_size=1, epochs=1)


test_data= scaled_data[training_data_len-90:,:]

x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(90, len(test_data)):
    x_test.append(test_data[i-90:i,0])


x_test=np.array(x_test)


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))


predictions= model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Standatd error measurement
rmse=np.sqrt(np.mean((predictions-y_test)**2))
print(rmse)


train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()

amazon_quote = web.DataReader('AMZN', data_source='yahoo', start='2019-01-01', end=today)
new_df=amazon_quote.filter(['Close'])

last_60_days = new_df[-90:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)

#Seven Day prediction, Predict stock price for future 7 days
apple_quote = web.DataReader('AMZN', data_source='yahoo', start='2019-01-01', end=today)
new_df=apple_quote.filter(['Close'])

last_60_days = new_df[-90:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)



last_60_days=new_df[-89:].values


firstprediction =np.array([pred_price])
last59 = numpy.append(last_60_days, firstprediction)
last59 = np.reshape(last59,(-1,1))

last59_scaled= scaler.transform(last59)
X_test2 = []
X_test2.append(last59_scaled)
X_test2 = np.array(X_test2)
X_test2=np.reshape(X_test2,(X_test2.shape[0], X_test2.shape[1],1))
pred_price2=model.predict(X_test2)
pred_price2=scaler.inverse_transform(pred_price2)

last_60_days=new_df[-88:].values
Secondprediction = np.array([pred_price2])
FirstSECprediction = numpy.append(firstprediction, Secondprediction)
last58 = numpy.append(last_60_days, FirstSECprediction)
last58 = np.reshape(last58,(-1,1))
last58_scaled= scaler.transform(last58)
X_test3 = []
X_test3.append(last58_scaled)
X_test3 = np.array(X_test3)
X_test3=np.reshape(X_test3,(X_test3.shape[0], X_test3.shape[1],1))
pred_price3=model.predict(X_test3)
pred_price3=scaler.inverse_transform(pred_price3)


last_60_days=new_df[-87:].values
Thirdprediction = np.array([pred_price3])
FirstThreeprediction = numpy.append(FirstSECprediction, Thirdprediction)
last57 = numpy.append(last_60_days, FirstThreeprediction)
last57 = np.reshape(last57,(-1,1))
last57_scaled= scaler.transform(last57)
X_test4 = []
X_test4.append(last57_scaled)
X_test4 = np.array(X_test4)
X_test4=np.reshape(X_test4,(X_test4.shape[0], X_test4.shape[1],1))
pred_price4=model.predict(X_test4)
pred_price4=scaler.inverse_transform(pred_price4)


last_60_days=new_df[-86:].values
Fourthprediction = np.array([pred_price4])
FirstFourthprediction = numpy.append(FirstThreeprediction, Fourthprediction)
last56 = numpy.append(last_60_days, FirstFourthprediction)
last56 = np.reshape(last56,(-1,1))
last56_scaled= scaler.transform(last56)
X_test5 = []
X_test5.append(last56_scaled)
X_test5 = np.array(X_test5)
X_test5=np.reshape(X_test5,(X_test5.shape[0], X_test5.shape[1],1))
pred_price5=model.predict(X_test5)
pred_price5=scaler.inverse_transform(pred_price5)

last_60_days=new_df[-85:].values
Fifthprediction = np.array([pred_price5])
FirstFifthprediction = numpy.append(FirstFourthprediction, Fifthprediction)
last55 = numpy.append(last_60_days, FirstFifthprediction)
last55 = np.reshape(last55,(-1,1))
last55_scaled= scaler.transform(last55)
X_test6 = []
X_test6.append(last55_scaled)
X_test6 = np.array(X_test6)
X_test6=np.reshape(X_test6,(X_test6.shape[0], X_test6.shape[1],1))
pred_price6=model.predict(X_test6)
pred_price6=scaler.inverse_transform(pred_price6)

last_60_days=new_df[-84:].values
Sixthprediction = np.array([pred_price6])
FirstSixthprediction = numpy.append(FirstFifthprediction, Sixthprediction)
last54 = numpy.append(last_60_days, FirstSixthprediction)
last54 = np.reshape(last54,(-1,1))
last54_scaled= scaler.transform(last54)
X_test7 = []
X_test7.append(last54_scaled)
X_test7 = np.array(X_test7)
X_test7=np.reshape(X_test7,(X_test7.shape[0], X_test7.shape[1],1))
pred_price7=model.predict(X_test7)
pred_price7=scaler.inverse_transform(pred_price7)

print("Price prediction for the next 7 trading days")

print(pred_price)
print(pred_price2)
print(pred_price3)
print(pred_price4)
print(pred_price5)
print(pred_price6)
print(pred_price7)


print("below is prediction with event bias correction")
#Reading the important index for further prediction
spdf = web.DataReader('^GSPC', data_source='yahoo', start=today, end=today)
new_spdf=spdf.filter(['Open'])
print("S&P 500 today")
print(new_spdf)
djdf = web.DataReader('^DJI', data_source='yahoo', start=today, end=today)
new_djdf=djdf.filter(['Open'])
print("DOW Jones Today")
print(new_djdf)
yesterday = today - timedelta(days=1)
spdfyesterday = web.DataReader('^GSPC', data_source='yahoo', start=yesterday, end=yesterday)
new_spdfyesterday=spdfyesterday.filter(['Open'])
print("S&P500 yesterday")
print(new_spdfyesterday)
djdfyesterday = web.DataReader('^DJI', data_source='yahoo', start=yesterday, end=yesterday)
new_djdfyesterday=djdfyesterday.filter(['Open'])
print("Dow Jones Yesterday")
print(new_djdfyesterday)
spv=new_spdf.values
spvy=new_spdfyesterday.values
spdifference=spv-spvy
print('SP500difference in 1 day')
print(spdifference)
djv=new_djdf.values
djvy=new_djdfyesterday.values
djdifference=djv-djvy
print('Dow Jones difference in 1 day')
print(djdifference)

apple_quote2 = web.DataReader('AMZN', data_source='yahoo', start='2019-01-01', end=today)
new_df2=apple_quote2.filter(['Close'])

last_60_days2 = new_df2[-90:].values
last_60_days_scaled2=scaler.transform(last_60_days2)
X_test12 = []
X_test12.append(last_60_days_scaled2)
X_test12 = np.array(X_test12)
X_test12=np.reshape(X_test12,(X_test12.shape[0], X_test12.shape[1],1))
pred_price12=model.predict(X_test12)
pred_price12=scaler.inverse_transform(pred_price12)

if spdifference < -200 and spdifference >-400:
    pred_price12=pred_price12*0.8
    print("significant drop in SP, turn down expectation")
elif spdifference < -400:
    print("Warning, abnormal activity in SP500 has been detected, turn down expectation")
    pred_price12 = pred_price12*0.7

if djdifference < -1000 and djdifference > -2000:
    pred_price12=pred_price12*0.8
    print("significant drop in Dow Jones, turn down expectation")
elif djdifference < -2000:
    print("Warning, abnormal activity in Dow Jones has been detected, turn down expectation")
    pred_price12 = pred_price12 * 0.7
last_60_days2=new_df2[-89:].values
firstprediction12 =np.array([pred_price12])
last592 = numpy.append(last_60_days2, firstprediction12)
last592 = np.reshape(last592,(-1,1))

last59_scaled12= scaler.transform(last592)
X_test212 = []
X_test212.append(last59_scaled12)
X_test212 = np.array(X_test212)
X_test212=np.reshape(X_test212,(X_test212.shape[0], X_test212.shape[1],1))

pred_price212=model.predict(X_test212)
pred_price212=scaler.inverse_transform(pred_price212)



#11
last_60_days2=new_df2[-88:].values
Secondprediction12 = np.array([pred_price212])
FirstSECprediction2 = numpy.append(firstprediction12, Secondprediction12)
last5812 = numpy.append(last_60_days2, FirstSECprediction2)
last5812 = np.reshape(last5812,(-1,1))
last58_scaled12= scaler.transform(last5812)
X_test312 = []
X_test312.append(last58_scaled12)
X_test312 = np.array(X_test312)
X_test312=np.reshape(X_test312,(X_test312.shape[0], X_test312.shape[1],1))
pred_price312=model.predict(X_test312)
pred_price312=scaler.inverse_transform(pred_price312)


last_60_days2=new_df2[-87:].values
Thirdprediction12 = np.array([pred_price312])
FirstThreeprediction2 = numpy.append(FirstSECprediction2, Thirdprediction12)
last5712 = numpy.append(last_60_days2, FirstThreeprediction2)
last5712 = np.reshape(last5712,(-1,1))
last57_scaled12= scaler.transform(last5712)
X_test412 = []
X_test412.append(last57_scaled12)
X_test412 = np.array(X_test412)
X_test412=np.reshape(X_test412,(X_test412.shape[0], X_test412.shape[1],1))
pred_price412=model.predict(X_test412)
pred_price412=scaler.inverse_transform(pred_price412)


last_60_days2=new_df2[-86:].values
Fourthprediction12 = np.array([pred_price412])
FirstFourthprediction2 = numpy.append(FirstThreeprediction2, Fourthprediction12)
last5612 = numpy.append(last_60_days2, FirstFourthprediction2)
last5612 = np.reshape(last5612,(-1,1))
last56_scaled12= scaler.transform(last5612)
X_test512 = []
X_test512.append(last56_scaled12)
X_test512 = np.array(X_test512)
X_test512=np.reshape(X_test512,(X_test512.shape[0], X_test512.shape[1],1))
pred_price512=model.predict(X_test512)
pred_price512=scaler.inverse_transform(pred_price512)

last_60_days2=new_df2[-85:].values
Fifthprediction12 = np.array([pred_price512])
FirstFifthprediction2 = numpy.append(FirstFourthprediction2, Fifthprediction12)
last5512 = numpy.append(last_60_days2, FirstFifthprediction2)
last5512 = np.reshape(last5512,(-1,1))
last55_scaled12= scaler.transform(last5512)
X_test612 = []
X_test612.append(last55_scaled12)
X_test612 = np.array(X_test612)
X_test612=np.reshape(X_test612,(X_test612.shape[0], X_test612.shape[1],1))
pred_price612=model.predict(X_test612)
pred_price612=scaler.inverse_transform(pred_price612)

last_60_days2=new_df2[-84:].values
Sixthprediction12 = np.array([pred_price612])
FirstSixthprediction2 = numpy.append(FirstFifthprediction2, Sixthprediction12)
last5412 = numpy.append(last_60_days2, FirstSixthprediction2)
last5412 = np.reshape(last5412,(-1,1))
last54_scaled12= scaler.transform(last5412)
X_test712 = []
X_test712.append(last54_scaled12)
X_test712 = np.array(X_test712)
X_test712=np.reshape(X_test712,(X_test712.shape[0], X_test712.shape[1],1))
pred_price712=model.predict(X_test712)
pred_price712=scaler.inverse_transform(pred_price712)





print("Price prediction for the next 7 trading days")

print(pred_price12)
print(pred_price212)
print(pred_price312)
print(pred_price412)
print(pred_price512)
print(pred_price612)
print(pred_price712)

