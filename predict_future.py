import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from threshold import get_threshold

TECHNICAL_DATA_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\technical_data_eurusd_h1.csv"
SAVE_PROD_MODEL_EURUSD_H1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\prob_model_eurusd_h1.h5"


class TrendPredictor:
    def __init__(self, data_path, model_path):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.look_back = 10
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.last_update_time = None
        self.close_price_index = None  # Індекс колонки з ціною закриття

    def load_and_prepare_data(self):
        """Завантаження та підготовка даних з часовими мітками"""
        df = pd.read_csv(self.data_path)

        # Перевіряємо наявність колонки Close
        if 'Close' not in df.columns:
            raise ValueError("У даних відсутня колонка 'Close' з цінами закриття")

        # Зберігаємо індекс колонки Close для подальшого використання
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.close_price_index = numeric_columns.get_loc('Close')

        # Додаємо часову мітку, якщо її немає
        if 'timestamp' not in df.columns:
            end_time = datetime.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)
            timestamps = [end_time - timedelta(hours=x) for x in range(len(df) - 1, -1, -1)]
            df['timestamp'] = timestamps

        # Конвертуємо у datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Сортуємо за часом
        df = df.sort_values('timestamp')

        self.last_update_time = df['timestamp'].max()
        return df

    def predict_future_trend(self, n_future_steps=10):
        """Прогнозування майбутніх значень з часовими мітками"""
        # Завантаження даних
        df = self.load_and_prepare_data()
        dataset = df.select_dtypes(include=[np.number]).values.astype('float32')

        # Зберігаємо оригінальні ціни закриття для масштабування
        original_close_prices = dataset[:, self.close_price_index]

        # Нормалізація
        dataset_scaled = self.scaler.fit_transform(dataset)

        # Завантаження моделі
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

        # Підготовка останніх даних
        last_sequence = dataset_scaled[-self.look_back:]
        future_predictions = []
        prediction_times = []

        # Генерація майбутніх часових міток
        current_time = self.last_update_time

        # Прогнозування
        current_sequence = last_sequence.reshape((1, self.look_back, dataset_scaled.shape[1]))

        for i in range(n_future_steps):
            # Прогноз наступного значення
            next_pred = self.model.predict(current_sequence, verbose=0)[0][0]

            # Додавання контрольованого шуму для реалістичності
            if i > 0:
                noise = np.random.normal(0, 0.0005)
                next_pred = np.clip(next_pred + noise, 0, 1)

            future_predictions.append(next_pred)

            # Генерація наступної часової мітки
            current_time += timedelta(hours=1)
            prediction_times.append(current_time)

            # Оновлення послідовності для наступного прогнозу
            next_features = self._generate_technical_features(
                current_sequence[0, -1],
                next_pred,
                self.close_price_index
            )
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = next_features

        # Перетворення прогнозів назад у початковий масштаб
        predictions_scaled = self._inverse_transform_predictions(
            future_predictions,
            original_close_prices
        )

        return pd.DataFrame({
            'timestamp': prediction_times,
            'predicted_close': predictions_scaled,
            'confidence': self._calculate_confidence_scores(future_predictions)
        })

    def _generate_technical_features(self, last_features, next_pred, close_index):
        """Генерація технічних індикаторів для наступного прогнозу"""
        feature_count = len(last_features)
        next_features = np.zeros(feature_count)
        next_features[close_index] = next_pred  # Використовуємо правильний індекс для ціни закриття

        # Оновлення технічних індикаторів з урахуванням їх взаємозв'язків
        for j in range(feature_count):
            if j != close_index:  # Пропускаємо ціну закриття, вона вже встановлена
                base_value = last_features[j]
                variation = np.random.normal(0, 0.0005)
                next_features[j] = np.clip(base_value + variation, 0, 1)

        return next_features

    def _inverse_transform_predictions(self, predictions, original_prices):
        """Перетворення прогнозів назад у початковий масштаб"""
        # Створюємо масив з усіма функціями
        extended = np.zeros((len(predictions), self.scaler.n_features_in_))
        extended[:, self.close_price_index] = predictions

        # Inverse transform
        full_inverse = self.scaler.inverse_transform(extended)

        # Повертаємо тільки ціни закриття
        return full_inverse[:, self.close_price_index]

    def _calculate_confidence_scores(self, predictions):
        """Розрахунок рівня впевненості для кожного прогнозу"""
        confidence_scores = []
        for i in range(len(predictions)):
            if i == 0:
                confidence = 0.95
            else:
                confidence = 0.95 * (0.98 ** i)
            confidence_scores.append(confidence)
        return confidence_scores

    def analyze_and_visualize(self, predictions_df, threshold_value):
        """Аналіз та візуалізація прогнозів з часовими мітками"""
        # Розрахунок трендів
        predictions_df['trend'] = 'Нейтральний'
        price_diff = predictions_df['predicted_close'].diff()
        predictions_df.loc[price_diff > threshold_value, 'trend'] = 'Зростання'
        predictions_df.loc[price_diff < -threshold_value, 'trend'] = 'Спадання'

        # Візуалізація
        plt.figure(figsize=(15, 8))

        # Графік цін закриття
        plt.subplot(2, 1, 1)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_close'],
                 marker='o', linestyle='-', linewidth=2, label='Прогноз ціни закриття')

        # Додавання кольорових міток для трендів
        for trend, color in zip(['Зростання', 'Спадання', 'Нейтральний'], ['green', 'red', 'blue']):
            mask = predictions_df['trend'] == trend
            if mask.any():
                plt.plot(predictions_df.loc[mask, 'timestamp'],
                         predictions_df.loc[mask, 'predicted_close'],
                         'o', color=color, label=trend)

        plt.title('Прогноз ціни закриття з часовими мітками')
        plt.xlabel('Час')
        plt.ylabel('Ціна закриття')
        plt.grid(True)
        plt.legend()

        # Графік впевненості
        plt.subplot(2, 1, 2)
        plt.plot(predictions_df['timestamp'], predictions_df['confidence'],
                 marker='o', linestyle='-', color='purple', label='Рівень впевненості')
        plt.title('Рівень впевненості прогнозу')
        plt.xlabel('Час')
        plt.ylabel('Впевненість')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        return predictions_df


def main():
    predictor = TrendPredictor(
        data_path=TECHNICAL_DATA_EURUSD_H1,
        model_path=SAVE_PROD_MODEL_EURUSD_H1
    )

    try:
        # Прогнозування
        n_future = 10
        predictions_df = predictor.predict_future_trend(n_future)

        # Отримання порогового значення для цін закриття
        df = pd.read_csv(TECHNICAL_DATA_EURUSD_H1)
        threshold = get_threshold(df["Close"])

        # Аналіз та візуалізація
        results_df = predictor.analyze_and_visualize(predictions_df, threshold)

        # Виведення статистики
        print("\nСтатистика прогнозів цін закриття:")
        print(f"Період прогнозування: з {results_df['timestamp'].min()} по {results_df['timestamp'].max()}")
        print("\nРозподіл трендів:")
        print(results_df['trend'].value_counts().to_string())
        print("\nСередня впевненість прогнозу:", f"{results_df['confidence'].mean():.2%}")
        print(
            f"\nДіапазон цін закриття: {results_df['predicted_close'].min():.4f} - {results_df['predicted_close'].max():.4f}")

    except Exception as e:
        print(f"Помилка при виконанні прогнозування: {str(e)}")
        raise


if __name__ == "__main__":
    main()