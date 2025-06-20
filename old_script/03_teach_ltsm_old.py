import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Шляхи до файлів
TECHNICAL_DATA_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\historic_data_eurusd_h1.csv"
SAVE_PROD_MODEL_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\historic_model_eurusd_h1.h5"
FILE_LOG = r"/log/historic_teach_ltsm_eurusd_m1.log"

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
logging.info("Модель почала навчатися")

# Завантаження та підготовка даних
dataframe = read_csv(TECHNICAL_DATA_EURUSD_H1, engine='python')
dataset = dataframe.values.astype('float32')

# Розділення на ціну та інші ознаки
price = dataset[:, 0].reshape(-1, 1)  # Цільова змінна (перший стовпець)
features = dataset[:, 1:]             # Інші 12 ознак

# Окреме масштабування
price_scaler = MinMaxScaler(feature_range=(0, 1))
feature_scaler = MinMaxScaler(feature_range=(0, 1))

price_scaled = price_scaler.fit_transform(price)
features_scaled = feature_scaler.fit_transform(features)

# Об'єднання масштабованих даних
dataset_scaled = np.hstack((price_scaled, features_scaled))

# Розділення на тренувальний та тестовий набори
train_size = int(len(dataset_scaled) * 0.8)
train, test = dataset_scaled[0:train_size], dataset_scaled[train_size:]

# Функція створення послідовностей
def create_dataset(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back, 0])  # Ціль - масштабована ціна
    return np.array(dataX), np.array(dataY)

look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Ресурсоємна операція - reshape
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 13))
testX = np.reshape(testX, (testX.shape[0], look_back, 13))

# Покращена архітектура моделі
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, 13), dropout=0.2),
    LSTM(50, dropout=0.2),
    Dense(1, activation='linear')  # Лінійна активація для регресії
])

model.compile(loss='mean_squared_error', optimizer='adam')

# Колбеки
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model_checkpoint = ModelCheckpoint(SAVE_PROD_MODEL_EURUSD_H1,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)

# Тренування
history = model.fit(
    trainX, trainY,
    epochs=100,
    batch_size=64,
    validation_data=(testX, testY),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Прогнозування
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Зворотнє масштабування тільки для ціни
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

# Виведення останніх 5 передбачень і реальних значень
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