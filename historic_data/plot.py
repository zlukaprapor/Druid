import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf

TECHNICAL_DATA_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\historic_data_eurusd_h1.csv"
SAVE_PROD_MODEL_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\historic_model_eurusd_h1.h5"


# Завантаження даних
dataframe = read_csv(TECHNICAL_DATA_EURUSD_H1, engine='python')
dataset = dataframe.values.astype('float32')


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
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Зміна форми даних
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

# Завантаження моделі
model = tf.keras.models.load_model(SAVE_PROD_MODEL_EURUSD_H1)

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

# Функція для знаходження верхньої межі порогу
def get_upper_threshold(close):
    difference = close.diff().abs()  # Різниці між цінами
    bins = pd.cut(difference, bins=10)  # Розбивка на 10 інтервалів
    bins = bins.value_counts().to_frame().reset_index()
    bins["index"] = bins["index"].apply(lambda x: x.right)  # Оновлення меж
    bins = bins.to_numpy()  # Перетворення в масив
    percentile_count = len(difference) * 0.85  # 85% від загальної кількості
    count = 0
    for i in range(10):
        count += bins[i, 1]
        if count > percentile_count:
            return bins[i, 0]  # Повертає верхню межу порогу

# Функція для обчислення ентропії
def get_entropy(labels, base=None):
    vc = pd.Series(labels).value_counts(normalize=True)
    base = math.e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()  # Розрахунок ентропії

# Функція для знаходження найкращого порогу
def get_threshold(close):
    difference = close.diff().drop(0).tolist()  # Різниці між цінами
    thres_upper_bound = get_upper_threshold(close)  # Верхня межа порогу
    best_entropy = -float('inf')  # Початкове значення ентропії
    threshold = 0
    temp_thres = 0
    while temp_thres < thres_upper_bound:
        labels = [2 if diff > temp_thres else 1 if -diff > temp_thres else 0 for diff in difference]
        entropy = get_entropy(labels)  # Розрахунок ентропії
        if entropy > best_entropy:
            best_entropy = entropy
            threshold = temp_thres
        temp_thres += 0.00001  # Збільшення порогу
    return threshold  # Повертає найкращий поріг

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
