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


def predict_future_trend(n_future_steps=10):
    """
    Виправлена функція для прогнозування майбутніх значень тренду
    """
    # Завантаження та підготовка даних
    try:
        dataframe = read_csv(TECHNICAL_DATA_EURUSD_M1, engine='python')
        dataset = dataframe.values.astype('float32')
    except Exception as e:
        print(f"Помилка при завантаженні даних: {str(e)}")
        raise

    # Видаляємо перші 20 рядків та перший стовпець
    dataset = dataset[20:]
    original_data = dataset.copy()
    dataset = dataset[:, 1:]

    # Нормалізуємо дані
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Параметри
    look_back = 5
    feature_count = dataset.shape[1]

    # Завантажуємо модель
    try:
        model = tf.keras.models.load_model(SAVE_PROD_MODEL_EURUSD_M1)
    except Exception as e:
        print(f"Помилка при завантаженні моделі: {str(e)}")
        raise

    # Беремо останні реальні дані
    last_real_data = dataset[-look_back:]
    future_predictions = []

    def inverse_transform(arr):
        """Функція для повернення нормалізованих даних до початкового масштабу"""
        extended = np.zeros((len(arr), dataset.shape[1]))
        extended[:, 0] = arr
        return scaler.inverse_transform(extended)[:, 0]

    # Отримуємо статистику змін з історичних даних
    historical_changes = np.diff(original_data[:, 0])
    std_dev = np.std(historical_changes)

    # Формуємо початкову послідовність
    current_sequence = last_real_data.reshape((1, look_back, feature_count))
    last_pred = current_sequence[0, -1, 0]

    for i in range(n_future_steps):
        # Отримуємо базовий прогноз
        next_pred = model.predict(current_sequence, verbose=0)[0][0]

        # Обмежуємо значення прогнозу
        next_pred = np.clip(next_pred, 0, 1)

        # Додаємо невелику випадковість, але контролюємо її межі
        if i > 0:
            noise = np.random.normal(0, 0.001)  # Зменшили рівень шуму
            next_pred = np.clip(next_pred + noise, 0, 1)  # Обмежуємо значення

        future_predictions.append(next_pred)

        # Створюємо наступну послідовність з обмеженням значень
        next_features = np.zeros(feature_count)
        next_features[0] = next_pred

        # Оновлюємо технічні індикатори з контрольованими значеннями
        for j in range(1, feature_count):
            base_value = current_sequence[0, -1, j]
            variation = np.random.normal(0, 0.001)
            next_features[j] = np.clip(base_value + variation, 0, 1)

        # Оновлюємо послідовність
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1] = next_features

    # Повертаємо значення до початкового масштабу
    future_predictions = np.array(future_predictions)
    future_predictions_scaled = inverse_transform(future_predictions)

    return future_predictions_scaled


def analyze_trend(predictions, threshold_value):
    """Аналіз напрямку тренду на основі прогнозів"""
    trends = []
    for i in range(1, len(predictions)):
        diff = predictions[i] - predictions[i - 1]
        if diff > threshold_value:
            trends.append("Зростання")
        elif -diff > threshold_value:
            trends.append("Спадання")
        else:
            trends.append("Нейтральний")
    return trends


def visualize_predictions(predictions, trends):
    """Візуалізація прогнозів та трендів"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(predictions)), predictions,
             marker='o', linestyle='-', linewidth=2, label='Прогноз')

    plt.title('Прогноз майбутнього тренду ціни')
    plt.xlabel('Часові кроки')
    plt.ylabel('Прогнозована ціна')
    plt.grid(True)
    plt.legend()

    # Додаємо позначки трендів
    for i, trend in enumerate(trends):
        color = 'green' if trend == "Зростання" else 'red' if trend == "Спадання" else 'blue'
        plt.annotate(trend,
                     (i + 1, predictions[i + 1]),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     color=color)

    plt.show()


# Головний код
if __name__ == "__main__":
    try:
        # Кількість майбутніх прогнозів
        n_future = 30

        print("Розпочинаємо прогнозування...")
        future_predictions = predict_future_trend(n_future)

        # Отримуємо порогове значення
        dataframe = read_csv(TECHNICAL_DATA_EURUSD_M1, engine='python')
        threshold = get_threshold(dataframe["Close"])

        # Аналізуємо тренди
        trends = analyze_trend(future_predictions, threshold)

        # Візуалізуємо результати
        visualize_predictions(future_predictions, trends)

        # Виводимо детальні результати
        print("\nДетальний аналіз прогнозів:")
        for i, (pred, trend) in enumerate(zip(future_predictions[1:], trends), 1):
            print(f"Крок {i}: Ціна = {pred:.4f}, Тренд: {trend}")

        # Виводимо загальну статистику
        trend_stats = pd.Series(trends).value_counts()
        print("\nСтатистика трендів:")
        for trend, count in trend_stats.items():
            print(f"{trend}: {count} разів ({count / len(trends) * 100:.1f}%)")

    except Exception as e:
        print(f"Виникла помилка: {str(e)}")
        raise