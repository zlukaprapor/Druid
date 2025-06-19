import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import pickle
import os
from datetime import datetime

tf.config.optimizer.set_jit(True)

# Шляхи до файлів
TECHNICAL_DATA_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\historic_data_eurusd_h1.csv"
SAVE_PROD_MODEL_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\historic_model_eurusd_h1.h5"
SAVE_FINETUNED_MODEL = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\finetuned_model_eurusd_h1.h5"
SCALERS_PATH = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\scalers.pkl"
LAST_TRAINING_INFO = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\last_training_info.pkl"
FILE_LOG = r"C:\Users\Oleksii\PycharmProjects\Druid\log\finetune_lstm_eurusd_h1.log"
BACKUP_DIR = r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\train_model\backups"

# Створення директорії для резервних копій
os.makedirs(BACKUP_DIR, exist_ok=True)

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(FILE_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class LSTMFineTuner:
    def __init__(self, model_path, scalers_path=None, training_info_path=None):
        self.model_path = model_path
        self.scalers_path = scalers_path
        self.training_info_path = training_info_path
        self.model = None
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 10
        self.last_training_info = {
            'last_data_size': 0,
            'last_training_date': None,
            'data_hash': None
        }

    def load_model_and_scalers(self):
        """Завантаження моделі та скейлерів"""
        try:
            self.model = load_model(self.model_path)
            logging.info(f"Модель завантажена з {self.model_path}")

            # Завантаження скейлерів
            if self.scalers_path and os.path.exists(self.scalers_path):
                with open(self.scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.price_scaler = scalers['price_scaler']
                    self.feature_scaler = scalers['feature_scaler']
                logging.info("Скейлери завантажені з файлу")
            else:
                logging.warning("Скейлери не знайдені, буде використано нове масштабування")

            # Завантаження інформації про останнє навчання
            if self.training_info_path and os.path.exists(self.training_info_path):
                with open(self.training_info_path, 'rb') as f:
                    self.last_training_info = pickle.load(f)
                logging.info("Інформація про останнє навчання завантажена")

        except Exception as e:
            logging.error(f"Помилка завантаження моделі: {e}")
            raise

    def save_scalers(self):
        """Збереження скейлерів"""
        scalers = {
            'price_scaler': self.price_scaler,
            'feature_scaler': self.feature_scaler
        }
        with open(self.scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        logging.info("Скейлери збережені")

    def save_training_info(self, data_size, data_hash):
        """Збереження інформації про навчання"""
        self.last_training_info = {
            'last_data_size': data_size,
            'last_training_date': datetime.now(),
            'data_hash': data_hash
        }
        if self.training_info_path:
            with open(self.training_info_path, 'wb') as f:
                pickle.dump(self.last_training_info, f)
            logging.info("Інформація про навчання збережена")

    def get_data_hash(self, data):
        """Отримання хешу даних для перевірки змін"""
        import hashlib
        return hashlib.md5(data.tobytes()).hexdigest()

    def check_if_retraining_needed(self, current_data):
        """Перевірка чи потрібно донавчання"""
        current_size = len(current_data)
        current_hash = self.get_data_hash(current_data)

        # Перевірка розміру даних
        size_increase = current_size - self.last_training_info['last_data_size']
        size_threshold = max(1000, self.last_training_info['last_data_size'] * 0.1)  # 10% або мінімум 1000 записів

        # Перевірка зміни даних
        data_changed = current_hash != self.last_training_info['data_hash']

        # Перевірка часу
        last_training = self.last_training_info.get('last_training_date')
        time_threshold_days = 7  # Донавчання раз на тиждень максимум
        time_passed = True
        if last_training:
            days_since_training = (datetime.now() - last_training).days
            time_passed = days_since_training >= time_threshold_days

        should_retrain = (size_increase >= size_threshold) and data_changed and time_passed

        logging.info(f"Поточний розмір даних: {current_size}")
        logging.info(f"Збільшення з останнього навчання: {size_increase}")
        logging.info(f"Поріг для донавчання: {size_threshold}")
        logging.info(f"Дані змінилися: {data_changed}")
        logging.info(f"Час з останнього навчання: {days_since_training if last_training else 'Ніколи'} днів")
        logging.info(f"Рекомендація донавчання: {should_retrain}")

        return should_retrain, size_increase

    def backup_model(self):
        """Створення резервної копії моделі"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}.h5")
        self.model.save(backup_path)
        logging.info(f"Резервна копія створена: {backup_path}")
        return backup_path

    def prepare_data(self, data_path, is_new_data=False):
        """Підготовка даних для навчання/донавчання"""
        dataframe = read_csv(data_path, engine='python')
        dataset = dataframe.values.astype('float32')

        # Розділення на ціну та ознаки
        price = dataset[:, 0].reshape(-1, 1)
        features = dataset[:, 1:]

        if is_new_data:
            # Для нових даних використовуємо існуючі скейлери
            price_scaled = self.price_scaler.transform(price)
            features_scaled = self.feature_scaler.transform(features)
        else:
            # Для старих даних підганяємо скейлери
            price_scaled = self.price_scaler.fit_transform(price)
            features_scaled = self.feature_scaler.fit_transform(features)

        dataset_scaled = np.hstack((price_scaled, features_scaled))
        return dataset_scaled, price, features

    def create_dataset(self, dataset, look_back=10):
        """Створення послідовностей для LSTM"""
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            dataX.append(dataset[i:(i + look_back)])
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def evaluate_model(self, X, y_true, description=""):
        """Оцінка моделі"""
        y_pred = self.model.predict(X, verbose=0)

        # Зворотнє масштабування
        y_pred_rescaled = self.price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_rescaled = self.price_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

        # Метрики
        rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
        mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
        mape = np.mean(np.abs((y_true_rescaled - y_pred_rescaled) / y_true_rescaled)) * 100
        r2 = r2_score(y_true_rescaled, y_pred_rescaled)

        logging.info(f"{description} Метрики:")
        logging.info(f"  RMSE: {rmse:.6f}")
        logging.info(f"  MAE: {mae:.6f}")
        logging.info(f"  MAPE: {mape:.6f}%")
        logging.info(f"  R²: {r2:.6f}")

        return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}

    def rolling_window_fine_tune(self, data_path, new_data_ratio=0.3,
                                 learning_rate=0.0001, epochs=30, batch_size=32,
                                 validation_split=0.2, patience=8):
        """Донавчання на нових даних з використанням rolling window підходу"""
        start_time = time.time()
        logging.info("Початок rolling window донавчання")

        # Завантаження всіх даних
        dataframe = read_csv(data_path, engine='python')
        dataset = dataframe.values.astype('float32')

        # Перевірка чи потрібно донавчання
        should_retrain, size_increase = self.check_if_retraining_needed(dataset)

        if not should_retrain:
            logging.info("Донавчання не потрібне на даний момент")
            return None, None

        # Створення резервної копії
        backup_path = self.backup_model()

        # Підготовка даних
        price = dataset[:, 0].reshape(-1, 1)
        features = dataset[:, 1:]

        # Використання існуючих скейлерів або створення нових
        try:
            # Спробуємо використати існуючі скейлери
            price_scaled = self.price_scaler.transform(price)
            features_scaled = self.feature_scaler.transform(features)
        except:
            # Якщо не вийшло, підганяємо скейлери заново
            logging.warning("Перепідгонка скейлерів на всіх даних")
            price_scaled = self.price_scaler.fit_transform(price)
            features_scaled = self.feature_scaler.fit_transform(features)

        dataset_scaled = np.hstack((price_scaled, features_scaled))

        # Стратегія rolling window: використовуємо останні дані для донавчання
        total_size = len(dataset_scaled)
        new_data_size = max(int(total_size * new_data_ratio), size_increase)

        # Беремо останні дані для донавчання
        recent_data = dataset_scaled[-new_data_size:]

        # Створення послідовностей
        X_new, y_new = self.create_dataset(recent_data, self.look_back)
        X_new = np.reshape(X_new, (X_new.shape[0], self.look_back, 13))

        logging.info(f"Використано останні {new_data_size} записів ({len(X_new)} зразків) для донавчання")

        # Оцінка моделі до донавчання
        initial_metrics = self.evaluate_model(X_new, y_new, "До донавчання")

        # Налаштування моделі для донавчання
        self.model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )

        # Колбеки
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                SAVE_FINETUNED_MODEL,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Донавчання
        history = self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Оцінка після донавчання
        final_metrics = self.evaluate_model(X_new, y_new, "Після донавчання")

        # Порівняння результатів
        self.compare_metrics(initial_metrics, final_metrics)

        # Збереження інформації про навчання
        self.save_training_info(total_size, self.get_data_hash(dataset))
        self.save_scalers()

        execution_time = (time.time() - start_time) / 60
        logging.info(f"Rolling window донавчання завершено за {execution_time:.2f} хв")

        return history, final_metrics

    def compare_metrics(self, initial_metrics, final_metrics):
        """Порівняння метрик до і після навчання"""
        logging.info("=== ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ===")
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            initial = initial_metrics[metric]
            final = final_metrics[metric]
            if metric == 'r2':
                improvement = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
            else:
                improvement = ((initial - final) / initial * 100) if initial != 0 else 0

            status = "✓" if improvement > 0 else "✗"
            logging.info(f"{status} {metric.upper()}: {initial:.6f} → {final:.6f} (зміна: {improvement:+.2f}%)")

    def smart_retrain_check(self, data_path, force_retrain=False):
        """Розумна перевірка і донавчання при необхідності"""
        if force_retrain:
            logging.info("Примусове донавчання")
            return self.rolling_window_fine_tune(data_path)

        # Завантаження даних для перевірки
        dataframe = read_csv(data_path, engine='python')
        dataset = dataframe.values.astype('float32')

        should_retrain, _ = self.check_if_retraining_needed(dataset)

        if should_retrain:
            return self.rolling_window_fine_tune(data_path)
        else:
            logging.info("Модель не потребує донавчання")
            return None, None

    def incremental_learning(self, old_data_path, new_data_path,
                             old_weight=0.7, new_weight=0.3, **kwargs):
        """Інкрементальне навчання з балансуванням старих і нових даних"""
        logging.info("Початок інкрементального навчання")

        # Підготовка старих даних
        old_dataset_scaled, _, _ = self.prepare_data(old_data_path, is_new_data=False)
        X_old, y_old = self.create_dataset(old_dataset_scaled, self.look_back)
        X_old = np.reshape(X_old, (X_old.shape[0], self.look_back, 13))

        # Підготовка нових даних
        new_dataset_scaled, _, _ = self.prepare_data(new_data_path, is_new_data=True)
        X_new, y_new = self.create_dataset(new_dataset_scaled, self.look_back)
        X_new = np.reshape(X_new, (X_new.shape[0], self.look_back, 13))

        # Семплювання для балансування
        old_sample_size = int(len(X_old) * old_weight)
        new_sample_size = int(len(X_new) * new_weight)

        # Випадковий вибір зразків
        old_indices = np.random.choice(len(X_old), min(old_sample_size, len(X_old)), replace=False)
        new_indices = np.random.choice(len(X_new), min(new_sample_size, len(X_new)), replace=False)

        # Об'єднання даних
        X_combined = np.vstack([X_old[old_indices], X_new[new_indices]])
        y_combined = np.hstack([y_old[old_indices], y_new[new_indices]])

        # Перемішування
        shuffle_indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[shuffle_indices]
        y_combined = y_combined[shuffle_indices]

        logging.info(f"Використано {len(old_indices)} старих і {len(new_indices)} нових зразків")

        # Донавчання на комбінованих даних
        return self.fine_tune_on_data(X_combined, y_combined, **kwargs)

    def fine_tune_on_data(self, X, y, learning_rate=0.0001, epochs=30,
                          batch_size=32, validation_split=0.2, patience=8):
        """Донавчання на підготовлених даних"""
        # Створення резервної копії
        backup_path = self.backup_model()

        # Налаштування оптимізатора
        self.model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )

        # Колбеки
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
            ModelCheckpoint(SAVE_FINETUNED_MODEL, monitor='val_loss', save_best_only=True)
        ]

        # Навчання
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def plot_training_history(self, history):
        """Візуалізація процесу навчання"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Fine-tuning')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss (log)')
        plt.plot(history.history['val_loss'], label='Validation Loss (log)')
        plt.yscale('log')
        plt.title('Model Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Основний код для виконання
def main():
    # Ініціалізація fine-tuner
    fine_tuner = LSTMFineTuner(
        SAVE_PROD_MODEL_EURUSD_H1,
        SCALERS_PATH,
        LAST_TRAINING_INFO
    )

    try:
        # Завантаження моделі
        fine_tuner.load_model_and_scalers()

        # Перевірка наявності даних
        if not os.path.exists(TECHNICAL_DATA_EURUSD_H1):
            logging.error(f"Файл з даними не знайдено: {TECHNICAL_DATA_EURUSD_H1}")
            return

        # Автоматична перевірка і донавчання при необхідності
        history, metrics = fine_tuner.smart_retrain_check(
            TECHNICAL_DATA_EURUSD_H1,
            force_retrain=False  # Встановіть True для примусового донавчання
        )

        if history is not None:
            # Візуалізація результатів
            fine_tuner.plot_training_history(history)
            logging.info("Донавчання успішно завершено!")
        else:
            logging.info("Донавчання не було виконано")

    except Exception as e:
        logging.error(f"Помилка під час донавчання: {e}")
        raise


# Функція для ручного донавчання з параметрами
def manual_retrain(new_data_ratio=0.3, learning_rate=0.0001, epochs=30):
    """Ручне донавчання з налаштуваними параметрами"""
    fine_tuner = LSTMFineTuner(
        SAVE_PROD_MODEL_EURUSD_H1,
        SCALERS_PATH,
        LAST_TRAINING_INFO
    )

    fine_tuner.load_model_and_scalers()

    history, metrics = fine_tuner.rolling_window_fine_tune(
        TECHNICAL_DATA_EURUSD_H1,
        new_data_ratio=new_data_ratio,
        learning_rate=learning_rate,
        epochs=epochs
    )

    if history:
        fine_tuner.plot_training_history(history)

    return history, metrics


if __name__ == "__main__":
    main()