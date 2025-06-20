import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os
import pickle
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPU пам'ять налаштовано на dynamic growth")
    except RuntimeError as e:
        logging.error(e)

# Шляхи до файлів
TECHNICAL_DATA_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\historic_data_eurusd_h1.csv"
SAVE_PROD_MODEL_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\historic_model_eurusd_h1.h5"
SCALER_PATH = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\scalers.pkl"
TRAINING_LOG_PATH = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\training_log.pkl"
FILE_LOG = r"C:\Users\Oleksii\PycharmProjects\Druid\log\historic_teach_ltsm_eurusd_m1.log"

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(FILE_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class IncrementalLSTMTrainer:
    def __init__(self, look_back=10, new_data_threshold=12):
        self.look_back = look_back
        self.new_data_threshold = new_data_threshold
        self.price_scaler = None
        self.feature_scaler = None
        self.model = None
        self.last_data_size = 0
        self.n_features = None  # Додаємо збереження кількості ознак

    def save_scalers(self):
        """Зберігає скейлери для майбутнього використання"""
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump({
                'price_scaler': self.price_scaler,
                'feature_scaler': self.feature_scaler,
                'n_features': self.n_features  # Зберігаємо кількість ознак
            }, f)
        logging.info(f"Скейлери збережено. Кількість ознак: {self.n_features}")

    def load_scalers(self):
        """Завантажує збережені скейлери"""
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scalers = pickle.load(f)
                self.price_scaler = scalers['price_scaler']
                self.feature_scaler = scalers['feature_scaler']
                self.n_features = scalers.get('n_features', None)
            logging.info(f"Скейлери завантажено. Кількість ознак: {self.n_features}")
            return True
        return False

    def save_training_log(self, data_size, metrics):
        """Зберігає інформацію про тренування"""
        log_data = {
            'last_data_size': data_size,
            'last_metrics': metrics,
            'timestamp': time.time(),
            'n_features': self.n_features
        }
        with open(TRAINING_LOG_PATH, 'wb') as f:
            pickle.dump(log_data, f)

    def load_training_log(self):
        """Завантажує інформацію про попереднє тренування"""
        if os.path.exists(TRAINING_LOG_PATH):
            with open(TRAINING_LOG_PATH, 'rb') as f:
                log_data = pickle.load(f)
                self.last_data_size = log_data.get('last_data_size', 0)
                if 'n_features' in log_data:
                    self.n_features = log_data['n_features']
                logging.info(f"Завантажено лог тренування. Останній розмір даних: {self.last_data_size}")
                return log_data
        return None

    def prepare_data(self, dataframe, is_incremental=False):
        """Підготовка даних з підтримкою інкрементального навчання"""
        # Видаляємо нечислові стовпці якщо є
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        dataframe_numeric = dataframe[numeric_columns].copy()

        # Видаляємо цільові змінні з ознак (якщо є)
        target_columns = [col for col in dataframe_numeric.columns if 'Target' in col or 'Direction' in col]
        feature_columns = [col for col in dataframe_numeric.columns if col not in target_columns]

        # Використовуємо тільки ознаки для навчання
        dataset = dataframe_numeric[feature_columns].values.astype('float32')

        logging.info(f"Розмір даних: {dataset.shape}")
        logging.info(f"Кількість ознак: {dataset.shape[1]}")

        # Розділення на ціну та інші ознаки
        price = dataset[:, 0].reshape(-1, 1)  # Перша колонка - Close
        features = dataset[:, 1:]  # Всі інші ознаки

        # Зберігаємо кількість ознак
        self.n_features = dataset.shape[1]

        if is_incremental and self.price_scaler is not None:
            # Для інкрементального навчання використовуємо існуючі скейлери
            logging.info("Використання існуючих скейлерів для інкрементального навчання")

            price_scaled = self.price_scaler.transform(price)
            features_scaled = self.feature_scaler.transform(features)

        else:
            # Повне навчання скейлерів
            self.price_scaler = MinMaxScaler(feature_range=(0, 1))
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

            price_scaled = self.price_scaler.fit_transform(price)
            features_scaled = self.feature_scaler.fit_transform(features)

            self.save_scalers()

        # Об'єднання масштабованих даних
        dataset_scaled = np.hstack((price_scaled, features_scaled))

        logging.info(f"Розмір масштабованих даних: {dataset_scaled.shape}")
        return dataset_scaled

    def create_dataset(self, dataset, look_back=None):
        """Створення послідовностей для LSTM"""
        if look_back is None:
            look_back = self.look_back

        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            dataX.append(dataset[i:(i + look_back)])
            dataY.append(dataset[i + look_back, 0])  # Цільова змінна - перша колонка (Close)

        X = np.array(dataX)
        Y = np.array(dataY)

        logging.info(f"Розмір X: {X.shape}, Розмір Y: {Y.shape}")
        return X, Y

    def create_model(self, input_shape):
        """Створення нової моделі"""
        logging.info(f"Створення моделі з input_shape: {input_shape}")

        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape, dropout=0.2),
            LSTM(50, dropout=0.2),
            Dense(1, activation='linear')
        ])

        # Використовуємо нижчий learning rate для інкрементального навчання
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model

    def train_incremental(self, new_data_X, new_data_Y, epochs=5):
        """Інкрементальне донавчання існуючої моделі"""
        logging.info("Початок інкрементального навчання")
        logging.info(f"Розмір нових даних: X={new_data_X.shape}, Y={new_data_Y.shape}")

        # Використовуємо менше епох і callback для раннього зупинення
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=2,
            verbose=1,
            restore_best_weights=True,
            min_delta=0.0001
        )

        # Навчання з малим learning rate і невеликою кількістю епох
        history = self.model.fit(
            new_data_X, new_data_Y,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1,
            shuffle=False
        )

        return history

    def train_full(self, dataset_scaled):
        """Повне навчання моделі з нуля"""
        logging.info("Початок повного навчання")

        # Розділення на тренувальний та тестовий набори
        train_size = int(len(dataset_scaled) * 0.8)
        train, test = dataset_scaled[0:train_size], dataset_scaled[train_size:]

        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        # Перевірка розмірів перед reshape
        logging.info(f"Перед reshape: trainX.shape={trainX.shape}")
        logging.info(f"Очікуваний розмір: ({trainX.shape[0]}, {self.look_back}, {self.n_features})")

        # Reshape для LSTM - використовуємо автоматично визначену кількість ознак
        trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, self.n_features))
        testX = np.reshape(testX, (testX.shape[0], self.look_back, self.n_features))

        logging.info(f"Після reshape: trainX.shape={trainX.shape}, testX.shape={testX.shape}")

        # Створення нової моделі з правильним input_shape
        input_shape = (self.look_back, self.n_features)
        self.model = self.create_model(input_shape)

        # Колбеки
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        model_checkpoint = ModelCheckpoint(
            SAVE_PROD_MODEL_EURUSD_H1,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        # Тренування
        history = self.model.fit(
            trainX, trainY,
            epochs=100,
            batch_size=64,
            validation_data=(testX, testY),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        return history, trainX, trainY, testX, testY

    def inverse_price_transform(self, scaled_price):
        """Зворотнє масштабування для ціни"""
        return self.price_scaler.inverse_transform(scaled_price.reshape(-1, 1)).flatten()

    def calculate_metrics(self, y_true, y_pred):
        """Розрахунок метрик"""

        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
        }


def main():
    start_time = time.time()

    # Ініціалізація тренера
    trainer = IncrementalLSTMTrainer()

    # Завантаження даних
    logging.info("Завантаження даних")
    dataframe = read_csv(TECHNICAL_DATA_EURUSD_H1, engine='python')
    current_data_size = len(dataframe)

    logging.info(f"Розмір завантажених даних: {dataframe.shape}")
    logging.info(f"Колонки: {list(dataframe.columns)}")

    # Перевірка чи потрібно інкрементальне навчання
    training_log = trainer.load_training_log()
    model_exists = os.path.exists(SAVE_PROD_MODEL_EURUSD_H1)
    scalers_loaded = trainer.load_scalers()

    # Визначення типу навчання
    if (model_exists and scalers_loaded and training_log and
            current_data_size - trainer.last_data_size >= trainer.new_data_threshold):

        logging.info(
            f"Виявлено {current_data_size - trainer.last_data_size} нових записів. Починаємо інкрементальне навчання")

        # Завантаження існуючої моделі
        trainer.model = load_model(SAVE_PROD_MODEL_EURUSD_H1)
        logging.info("Модель завантажено")

        # Підготовка даних для інкрементального навчання
        dataset_scaled = trainer.prepare_data(dataframe, is_incremental=True)

        # Беремо тільки нові дані для донавчання
        new_data = dataset_scaled[-trainer.new_data_threshold - trainer.look_back - 1:]
        new_X, new_Y = trainer.create_dataset(new_data)
        new_X = np.reshape(new_X, (new_X.shape[0], trainer.look_back, trainer.n_features))

        # Інкрементальне навчання
        history = trainer.train_incremental(new_X, new_Y, epochs=3)

        # Збереження оновленої моделі
        trainer.model.save(SAVE_PROD_MODEL_EURUSD_H1)

        # Тестування на останніх даних
        test_data = dataset_scaled[-50:]
        testX, testY = trainer.create_dataset(test_data)
        testX = np.reshape(testX, (testX.shape[0], trainer.look_back, trainer.n_features))

        testPredict = trainer.model.predict(testX)
        testPredict = trainer.inverse_price_transform(testPredict)
        testY = trainer.inverse_price_transform(testY.reshape(-1, 1))

        # Розрахунок метрик
        metrics = trainer.calculate_metrics(testY, testPredict)

        logging.info("Інкрементальне навчання завершено")

    else:
        logging.info("Початок повного навчання моделі")

        # Підготовка даних
        dataset_scaled = trainer.prepare_data(dataframe, is_incremental=False)

        # Повне навчання
        history, trainX, trainY, testX, testY = trainer.train_full(dataset_scaled)

        # Прогнозування
        trainPredict = trainer.model.predict(trainX)
        testPredict = trainer.model.predict(testX)

        # Зворотнє масштабування
        trainPredict = trainer.inverse_price_transform(trainPredict)
        testPredict = trainer.inverse_price_transform(testPredict)
        trainY = trainer.inverse_price_transform(trainY.reshape(-1, 1))
        testY = trainer.inverse_price_transform(testY.reshape(-1, 1))

        # Розрахунок метрик
        metrics = trainer.calculate_metrics(testY, testPredict)

        # Візуалізація
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(trainPredict, color='green', alpha=0.5, label='Train Predict')
            plt.plot(np.arange(len(trainPredict), len(trainPredict) + len(testPredict)),
                     testPredict, color='blue', label='Test Predict')
            plt.plot(np.arange(len(trainY)), trainY, color='orange', alpha=0.5, label='True Train')
            plt.plot(np.arange(len(trainY), len(trainY) + len(testY)),
                     testY, color='red', alpha=0.5, label='True Test')
            plt.legend()
            plt.title('Forecast vs Reality')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.show()
        except Exception as e:
            logging.warning(f"Помилка при створенні графіку: {e}")

    # Виведення метрик
    for name, value in metrics.items():
        logging.info(f"{name}: {value:.6f}" + ("%" if name == 'MAPE' else ""))

    # Збереження логу тренування
    trainer.save_training_log(current_data_size, metrics)

    # Час виконання
    execution_time = (time.time() - start_time) / 60
    logging.info(f"Час виконання: {execution_time:.2f} хв")


if __name__ == "__main__":
    main()