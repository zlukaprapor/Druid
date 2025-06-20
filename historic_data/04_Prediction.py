import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os
import pickle
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Шляхи до файлів
TECHNICAL_DATA_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\historic_data_eurusd_h1.csv"
SAVE_PROD_MODEL_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\historic_model_eurusd_h1.h5"
SCALER_PATH = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\scalers.pkl"
FILE_LOG = r"C:\Users\Oleksii\PycharmProjects\Druid\log\04_Prediction_log.log"

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(FILE_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

start_time = time.time()
logging.info("Початок прогнозування")

# Завантаження скейлерів та кількості ознак
with open(SCALER_PATH, 'rb') as f:
    scalers = pickle.load(f)
    price_scaler = scalers['price_scaler']
    feature_scaler = scalers['feature_scaler']
    n_features = scalers['n_features']  # Має бути 89

# Завантаження CSV
dataframe = read_csv(TECHNICAL_DATA_EURUSD_H1, engine='python')
numeric_data = dataframe.select_dtypes(include=[np.number])

# Використовуємо лише перші n_features колонок (щоб узгодити з тренуванням)
data_used = numeric_data.iloc[:, :n_features].values.astype('float32')

# Розділення на ціну та ознаки
price = data_used[:, 0].reshape(-1, 1)
features = data_used[:, 1:]

# Масштабування
price_scaled = price_scaler.transform(price)
features_scaled = feature_scaler.transform(features)
dataset_scaled = np.hstack((price_scaled, features_scaled))

# Параметри
look_back = 10

# Функція створення послідовностей
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Розбиття на train/test
train_size = int(len(dataset_scaled) * 0.8)
train, test = dataset_scaled[0:train_size], dataset_scaled[train_size:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], look_back, n_features))
testX = np.reshape(testX, (testX.shape[0], look_back, n_features))

# Завантаження моделі
model = load_model(SAVE_PROD_MODEL_EURUSD_H1)
logging.info("Модель завантажено")

# Прогнозування
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Зворотнє масштабування ціни
def inverse_price_transform(scaled_price):
    return price_scaler.inverse_transform(scaled_price.reshape(-1, 1)).flatten()

trainPredict = inverse_price_transform(trainPredict)
testPredict = inverse_price_transform(testPredict)
trainY = inverse_price_transform(trainY.reshape(-1, 1))
testY = inverse_price_transform(testY.reshape(-1, 1))

# Метрики
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics = {
    'RMSE': np.sqrt(mean_squared_error(testY, testPredict)),
    'MAE': mean_absolute_error(testY, testPredict),
    'MAPE': mean_absolute_percentage_error(testY, testPredict),
    'R²': r2_score(testY, testPredict),
}

for name, value in metrics.items():
    logging.info(f"{name}: {value:.6f}" + ("%" if name == 'MAPE' else ""))

logging.info(f"Test Predictions (Last): {testPredict[-1:]}")
logging.info(f"Actual Values (Last): {testY[-1:]}")

# Візуалізація
plt.figure(figsize=(14, 7))
plt.plot(trainPredict, color='green', alpha=0.5, label='Train Predict')
plt.plot(np.arange(len(trainPredict), len(trainPredict)+len(testPredict)),
         testPredict, color='blue', label='Test Predict')
plt.plot(np.arange(len(trainY)), trainY, color='orange', alpha=0.5, label='True Train')
plt.plot(np.arange(len(trainY), len(trainY)+len(testY)),
         testY, color='red', alpha=0.5, label='True Test')
plt.legend()
plt.title('Forecast vs Reality')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.show()

# Час виконання
execution_time = (time.time() - start_time) / 60
logging.info(f"Час виконання: {execution_time:.2f} хв")
