import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from threshold import get_threshold
import tensorflow as tf

TECHNICAL_DATA_EURUSD_M1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\technical_data_eurusd_m1.csv"
SAVE_PROD_MODEL_EURUSD_M1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\prob_model_eurusd_m1.h5"


# Завантаження даних
dataframe = read_csv(TECHNICAL_DATA_EURUSD_M1, engine='python')
dataset = dataframe.values.astype('float32')

# Відкидаємо перші 20 рядків
dataset = dataset[20:]
dataset = dataset[:, 1:]

# Нормалізація даних
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Розподіл даних на тренувальний та тестовий набори
train_size = int(len(dataset) * 0.8)
train, test = dataset[:train_size, :], dataset[train_size:, :]

# Функція створення датасетів для моделі
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Визначення параметра look_back
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Зміна форми даних
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

# Завантаження моделі
model = tf.keras.models.load_model(SAVE_PROD_MODEL_EURUSD_M1)

# Прогнозування
trainPredict = np.squeeze(model.predict(trainX))
testPredict = np.squeeze(model.predict(testX))

# Функція інверсної трансформації
def inverse_transform(arr):
    extended = np.zeros((len(arr), dataset.shape[1]))
    extended[:, 0] = arr
    return scaler.inverse_transform(extended)[:, 0]

# Зворотнє масштабування
trainPredict = inverse_transform(trainPredict)
testPredict = inverse_transform(testPredict)
trainY = inverse_transform(trainY)
testY = inverse_transform(testY)

# Обчислення порогу та прийняття рішень
threshold = get_threshold(dataframe["Close"])

def decision(diff):
    if diff > threshold:
        return "I"
    if -diff > threshold:
        return "D"
    else:
        return "N"

def union(a, b):
    if a == "I" and b == "I":
        return "C"
    if a == "D" and b == "D":
        return "C"
    if a == "N" or b == "N":
        return "N"
    else:
        return "F"

# Створення таблиці результатів
table = pd.DataFrame([testY, testPredict]).transpose()
table.columns = ["actual", "predict"]
table["p-a"] = table.predict - table.actual
table["decision"] = table["p-a"].apply(decision)
table["correct"] = table["actual"].diff().shift(-1).apply(
    lambda x: "I" if x > 0 else "D" if x < 0 else "N"
)
table["union"] = table.apply(lambda x: union(x["decision"], x["correct"]), axis=1)
counts = table["union"].value_counts()
frequency = table["union"].value_counts(normalize=True)
print(pd.DataFrame({"counts": counts, "frequency": frequency}))

# Зміщення передбачень для узгодження
testPredict = np.delete(testPredict, -1)
testY = np.delete(testY, 0)

# Візуалізація результатів
plt.plot(testPredict, color="blue", label="Predictions")
plt.plot(testY, color="red", label="Actual")
plt.legend()
plt.show()

# RMSE оцінка
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print(f'Test Score: {testScore:.6f} RMSE')
