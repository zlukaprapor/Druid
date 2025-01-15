import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Activation, Dense, LSTM
from tensorflow.python.keras.models import Sequential

# Шляхи до файлів з даними та для збереження моделі
TECHNICAL_DATA_EURUSD_M1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\technical_data_eurusd_m1.csv"
SAVE_PROD_MODEL_EURUSD_M1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\prob_model_eurusd_m1.h5"
FILE_LOG = r"C:\Users\Oleksii\PycharmProjects\Druid\log\model_eurusd_m1.log"

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(FILE_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Початок вимірювання часу
start_time = time.time()
logging.info("Модель почала навчатися")

# Завантаження даних з CSV-файлу
dataframe = read_csv(TECHNICAL_DATA_EURUSD_M1, engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')  # Перетворення даних у формат float32

# Обрізання перших 20 значень і видалення першого стовпця (заголовки)
#dataset = dataset[30:]  # Видалення перших 20 рядків
#dataset = dataset[:, 1:]  # Видалення першого стовпця

# Нормалізація даних у діапазоні [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Розділення даних на навчальну та тестову вибірки (80% - навчання, 20% - тестування)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Функція для створення вибірки з використанням затримки (look_back)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]  # Вибірка з look_back значень
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # Цільове значення (перший стовпець)
    return np.array(dataX), np.array(dataY)

# Кількість попередніх кроків для прогнозу
look_back = 5
trainX, trainY = create_dataset(train, look_back)  # Навчальна вибірка
testX, testY = create_dataset(test, look_back)  # Тестова вибірка

# Зміна форми даних для моделі LSTM [зразки, часові кроки, ознаки]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 13))
testX = np.reshape(testX, (testX.shape[0], look_back, 13))

# Створення моделі LSTM
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 13)))  # Перший LSTM шар з return_sequences=True
model.add(LSTM(50))  # Другий LSTM шар
model.add(Dense(1))  # Вихідний шар (1 нейрон)
model.add(Activation('sigmoid'))  # Активаційна функція
model.compile(loss='mean_squared_error', optimizer='adam')  # Налаштування моделі

# Навчання моделі
model.fit(trainX, trainY, epochs=30, batch_size=256, verbose=1)

# Збереження навченої моделі
model.save(SAVE_PROD_MODEL_EURUSD_M1)

# Передбачення на навчальній і тестовій вибірках
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Перетворення передбачень у вигляд одновимірних масивів
trainPredict = np.squeeze(trainPredict)
testPredict = np.squeeze(testPredict)

# Функція для зворотного перетворення нормалізованих даних
def inverse_transform(arr):
    extended = np.zeros((len(arr), 13))  # Розширення масиву до 9 стовпців
    extended[:, 0] = arr  # Вставка даних у перший стовпець
    return scaler.inverse_transform(extended)[:, 0]  # Зворотна нормалізація

# Зворотне перетворення передбачень і цільових значень
trainPredict = inverse_transform(trainPredict)
testPredict = inverse_transform(testPredict)
trainY = inverse_transform(trainY)
testY = inverse_transform(testY)

# Обрізання тестових даних для відповідності прогнозам
testPredict = np.delete(testPredict, -1)  # Видалення останнього значення з передбачень
testY = np.delete(testY, 0)  # Видалення першого значення з реальних даних

# Побудова графіків для порівняння передбачень і реальних значень
plt.plot(testPredict, color="blue", label="Predictions")
plt.plot(testY, color="red", label="Actual")
plt.legend()
plt.show()

# Обчислення метрик RMSE та MAE
testScore = np.sqrt(mean_squared_error(testY, testPredict))
scaled_rmse = np.sqrt(mean_squared_error(testY, testPredict))
original_rmse = np.sqrt(mean_squared_error(inverse_transform(testY), inverse_transform(testPredict)))
scaled_mae = mean_absolute_error(testY, testPredict)
original_mae = mean_absolute_error(inverse_transform(testY), inverse_transform(testPredict))

# Виведення метрик
print(f'Test Score: {testScore:.6f} RMSE')
print("Scaled RMSE:", scaled_rmse)
print("Original RMSE:", original_rmse)
print("Scaled MAE:", scaled_mae)
print("Original MAE:", original_mae)

# Виведення останніх 5 передбачень і реальних значень
print("Test Predictions (Last 5):", testPredict[-5:])
print("Actual Values (Last 5):", testY[-5:])

# Кінець вимірювання часу
end_time = time.time()
execution_time = (end_time - start_time) / 60  # Переводимо в хвилини
logging.info(f"Модель навчилася. Загальний час: {execution_time:.2f} хвилин")