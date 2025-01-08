import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential

TECHNICAL_DATA_EURUSDM1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\technical_data_eurusd.csv"
SAVE_PROD_MODEL = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\prob_model.h5"

dataframe = read_csv(TECHNICAL_DATA_EURUSDM1, engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# cut off first 20 values
dataset = dataset[20:]
dataset = dataset[:, 1:]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# convert an array of values into a dataset matrix


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back,9))
testX = np.reshape(testX, (testX.shape[0], look_back,9))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 9)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=256, verbose=1)
model.save(SAVE_PROD_MODEL)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = np.squeeze(trainPredict)
testPredict = np.squeeze(testPredict)


def inverse_transform(arr):
    extended = np.zeros((len(arr), 9))
    extended[:, 0] = arr
    return scaler.inverse_transform(extended)[:, 0]


trainPredict = inverse_transform(trainPredict)
testPredict = inverse_transform(testPredict)
trainY = inverse_transform(trainY)
testY = inverse_transform(testY)

# shift predictions up by one
testPredict = np.delete(testPredict, -1)
testY = np.delete(testY, 0)

plt.plot(testPredict, color="blue")
plt.plot(testY, color="red")
plt.show()

testScore = np.sqrt(mean_squared_error(testY, testPredict))
scaled_rmse = np.sqrt(mean_squared_error(testY, testPredict))
original_rmse = np.sqrt(mean_squared_error(inverse_transform(testY), inverse_transform(testPredict)))
scaled_mae = mean_absolute_error(testY, testPredict)
original_mae = mean_absolute_error(inverse_transform(testY), inverse_transform(testPredict))

print(f'Test Score: {testScore:.6f} RMSE')
print("Scaled RMSE:", scaled_rmse)
print("Original RMSE:", original_rmse)
print("Scaled MAE:", scaled_mae)
print("Original MAE:", original_mae)
print("Test Predictions:", testPredict[-5:])
print("Actual Values:", testY[-5:])
