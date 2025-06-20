import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Константи
EURUSD_H1 = Path(r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\EURUSDH1.csv")
TECHNICAL_DATA_EURUSD_H1 = Path(r"C:\Users\Oleksii\PycharmProjects\Druid\historic_data\csv\historic_data_eurusd_h1.csv")
ROWS_TO_SKIP = 50  # Збільшено для більш стабільних індикаторів

# Параметри технічних індикаторів
BOLLINGER_PERIOD = 20
RSI_PERIOD = 14  # Стандартний період для RSI
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ROC_PERIOD = 10
MOMENTUM_PERIOD = 10
CCI_PERIOD = 20
ATR_PERIOD = 14
STOCH_K = 14
STOCH_D = 3


def load_data(file_path: Path) -> pd.DataFrame:
    """Завантаження та початкова обробка даних."""
    try:
        df = pd.read_csv(file_path)

        # Перевірка наявності стовпця Date
        if 'Date' not in df.columns:
            print("Помилка: Стовпець 'Date' не знайдено")
            sys.exit(1)

        # Конвертування Date в datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        return df
    except FileNotFoundError:
        print(f"Помилка: Файл {file_path} не знайдено")
        sys.exit(1)
    except Exception as e:
        print(f"Помилка при завантаженні даних: {str(e)}")
        sys.exit(1)


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Додавання часових ознак для форекс торгівлі."""
    result = data.copy()

    # Основні часові ознаки
    result['Hour'] = data['Date'].dt.hour
    result['DayOfWeek'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    result['DayOfMonth'] = data['Date'].dt.day
    result['Month'] = data['Date'].dt.month
    result['Quarter'] = data['Date'].dt.quarter

    # Форекс-специфічні сесії (GMT час)
    def get_forex_session(hour, day_of_week):
        """Визначення торгової сесії"""
        # Вихідні (субота-неділя частково)
        if day_of_week == 5 and hour >= 22:  # П'ятниця після 22:00
            return 0  # Закриття
        if day_of_week == 6:  # Субота
            return 0
        if day_of_week == 0 and hour < 22:  # Неділя до 22:00
            return 0

        # Торгові сесії
        if 22 <= hour or hour < 7:  # 22:00-07:00 GMT (Sydney + Asian)
            return 1  # Азіатська сесія
        elif 7 <= hour < 15:  # 07:00-15:00 GMT
            return 2  # Європейська сесія
        elif 15 <= hour < 22:  # 15:00-22:00 GMT
            return 3  # Американська сесія

        return 1

    result['ForexSession'] = result.apply(
        lambda row: get_forex_session(row['Hour'], row['DayOfWeek']), axis=1
    )

    # Циклічне кодування часових ознак (важливо для нейронних мереж)
    result['Hour_sin'] = np.sin(2 * np.pi * result['Hour'] / 24)
    result['Hour_cos'] = np.cos(2 * np.pi * result['Hour'] / 24)
    result['DayOfWeek_sin'] = np.sin(2 * np.pi * result['DayOfWeek'] / 7)
    result['DayOfWeek_cos'] = np.cos(2 * np.pi * result['DayOfWeek'] / 7)
    result['Month_sin'] = np.sin(2 * np.pi * result['Month'] / 12)
    result['Month_cos'] = np.cos(2 * np.pi * result['Month'] / 12)

    return result


def calculate_advanced_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Розрахунок розширених технічних індикаторів."""
    result = data.copy()

    # Основні ціни
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # === ТРЕНДОВІ ІНДИКАТОРИ === #

    # Multiple Moving Averages
    result['SMA_5'] = close.rolling(5).mean()
    result['SMA_10'] = close.rolling(10).mean()
    result['SMA_20'] = close.rolling(20).mean()
    result['SMA_50'] = close.rolling(50).mean()

    # Exponential Moving Averages
    result['EMA_12'] = close.ewm(span=12).mean()
    result['EMA_26'] = close.ewm(span=26).mean()

    # MACD з сигнальною лінією
    result['MACD'] = result['EMA_12'] - result['EMA_26']
    result['MACD_Signal'] = result['MACD'].ewm(span=MACD_SIGNAL).mean()
    result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']

    # === ІНДИКАТОРИ ВОЛАТИЛЬНОСТІ === #

    # Bollinger Bands
    bb_ma = close.rolling(BOLLINGER_PERIOD).mean()
    bb_std = close.rolling(BOLLINGER_PERIOD).std()
    result['BB_Upper'] = bb_ma + 2 * bb_std
    result['BB_Lower'] = bb_ma - 2 * bb_std
    result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / bb_ma
    result['BB_Position'] = (close - result['BB_Lower']) / (result['BB_Upper'] - result['BB_Lower'])

    # Average True Range (ATR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = true_range.rolling(ATR_PERIOD).mean()

    # Volatility (стандартне відхилення returns)
    returns = close.pct_change()
    result['Volatility'] = returns.rolling(20).std()

    # === ОСЦИЛЯТОРИ === #

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    lowest_low = low.rolling(STOCH_K).min()
    highest_high = high.rolling(STOCH_K).max()
    result['Stoch_K'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    result['Stoch_D'] = result['Stoch_K'].rolling(STOCH_D).mean()

    # CCI (Commodity Channel Index)
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(CCI_PERIOD).mean()
    mad = typical_price.rolling(CCI_PERIOD).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    result['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

    # Williams %R
    result['Williams_R'] = -100 * (highest_high - close) / (highest_high - lowest_low)

    # === ІНДИКАТОРИ МОМЕНТУМУ === #

    # Rate of Change (ROC)
    result['ROC'] = ((close - close.shift(ROC_PERIOD)) / close.shift(ROC_PERIOD)) * 100

    # Momentum
    result['Momentum'] = close - close.shift(MOMENTUM_PERIOD)

    # Price Rate of Change
    result['PROC'] = close.pct_change(periods=10) * 100

    # === ОБСЯГОВІ ІНДИКАТОРИ === #

    # Volume Rate of Change
    result['Volume_ROC'] = volume.pct_change(periods=10) * 100

    # Volume Moving Average
    result['Volume_MA'] = volume.rolling(20).mean()
    result['Volume_Ratio'] = volume / result['Volume_MA']

    # === ДОДАТКОВІ ОЗНАКИ === #

    # Price returns
    result['Returns_1'] = close.pct_change(1)
    result['Returns_5'] = close.pct_change(5)
    result['Returns_10'] = close.pct_change(10)

    # High-Low spread
    result['HL_Spread'] = (high - low) / close

    # Open-Close spread
    result['OC_Spread'] = (close - data['Open']) / data['Open']

    # Price position within the day's range
    result['Price_Position'] = (close - low) / (high - low)

    # Gaps
    result['Gap'] = (data['Open'] - close.shift(1)) / close.shift(1)

    # Support/Resistance levels (simplified)
    result['Support'] = low.rolling(20).min()
    result['Resistance'] = high.rolling(20).max()
    result['Support_Distance'] = (close - result['Support']) / close
    result['Resistance_Distance'] = (result['Resistance'] - close) / close

    return result


def add_lagged_features(data: pd.DataFrame, lags=[1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Додавання лагових ознак."""
    result = data.copy()

    # Основні ціни з лагами
    for lag in lags:
        result[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        result[f'Returns_lag_{lag}'] = data['Close'].pct_change().shift(lag)
        result[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)

    return result


def add_rolling_statistics(data: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
    """Додавання ковзних статистик."""
    result = data.copy()

    returns = data['Close'].pct_change()

    for window in windows:
        # Ковзні статистики для returns
        result[f'Returns_mean_{window}'] = returns.rolling(window).mean()
        result[f'Returns_std_{window}'] = returns.rolling(window).std()
        result[f'Returns_skew_{window}'] = returns.rolling(window).skew()
        result[f'Returns_kurt_{window}'] = returns.rolling(window).kurt()

        # Ковзні статистики для volume
        result[f'Volume_mean_{window}'] = data['Volume'].rolling(window).mean()
        result[f'Volume_std_{window}'] = data['Volume'].rolling(window).std()

        # Min/Max за період
        result[f'High_max_{window}'] = data['High'].rolling(window).max()
        result[f'Low_min_{window}'] = data['Low'].rolling(window).min()

    return result


def validate_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Валідація та очищення даних."""
    print(f"Розмір даних до очищення: {data.shape}")

    # Заміна нескінченних значень на NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    # Видалення стовпців з занадто великою кількістю NaN
    threshold = 0.3  # Максимум 30% пропущених значень
    cols_to_drop = []
    for col in data.columns:
        if data[col].isna().sum() / len(data) > threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Видалено стовпці з великою кількістю NaN: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)

    # Заповнення пропущених значень
    # Форвард-філ для більшості стовпців
    data = data.fillna(method='ffill')

    # Backwards-філ для залишкових NaN
    data = data.fillna(method='bfill')

    # Видалення рядків, які все ще мають NaN
    data = data.dropna()

    print(f"Розмір даних після очищення: {data.shape}")

    # Перевірка на викиди (Z-score)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std())
    outliers = (z_scores > 5).any(axis=1)

    if outliers.sum() > 0:
        print(f"Виявлено {outliers.sum()} викидів")
        # Можна видалити або обрізати викиди
        # data = data[~outliers]  # Розкоментувати для видалення викидів

    return data


def create_target_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Створення цільових змінних для різних стратегій."""
    result = data.copy()

    # Майбутні returns
    result['Target_1h'] = data['Close'].shift(-1) / data['Close'] - 1
    result['Target_4h'] = data['Close'].shift(-4) / data['Close'] - 1
    result['Target_24h'] = data['Close'].shift(-24) / data['Close'] - 1

    # Бінарні цілі (напрямок)
    result['Direction_1h'] = (result['Target_1h'] > 0).astype(int)
    result['Direction_4h'] = (result['Target_4h'] > 0).astype(int)

    # Цілі для класифікації волатільності
    result['High_Volatility'] = (data['Close'].pct_change().rolling(24).std() >
                                 data['Close'].pct_change().rolling(100).std().rolling(50).mean()).astype(int)

    return result


def save_processed_data(data: pd.DataFrame, file_path: Path) -> None:
    """Збереження оброблених даних."""
    try:
        # Видалення оригінальних стовпців Date для моделі
        columns_to_save = [col for col in data.columns if
                           col not in ['Date', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter']]

        # Перестановка стовпців - Close на перше місце
        if 'Close' in columns_to_save:
            columns_to_save.remove('Close')
            columns_to_save = ['Close'] + columns_to_save

        final_data = data[columns_to_save]

        final_data.to_csv(file_path, index=False)
        print(f"\nДані успішно збережено у файл: {file_path}")
        print(f"Кількість ознак: {len(columns_to_save)}")
        print(f"Кількість записів: {len(final_data)}")

    except Exception as e:
        print(f"Помилка при збереженні даних: {str(e)}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("РОЗШИРЕНА ПІДГОТОВКА ДАНИХ ДЛЯ ФОРЕКС ПРОГНОЗУВАННЯ")
    print("=" * 60)

    # Завантаження сирих даних
    print("\n1. Завантаження даних...", end=' ')
    df = load_data(EURUSD_H1)
    print(f"OK ({len(df)} записів)")

    # Додавання часових ознак
    print("2. Додавання часових ознак...", end=' ')
    df = add_time_features(df)
    print("OK")

    # Розрахунок технічних індикаторів
    print("3. Розрахунок технічних індикаторів...", end=' ')
    df = calculate_advanced_indicators(df)
    print("OK")

    # Додавання лагових ознак
    print("4. Додавання лагових ознак...", end=' ')
    df = add_lagged_features(df)
    print("OK")

    # Ковзні статистики
    print("5. Розрахунок ковзних статистик...", end=' ')
    df = add_rolling_statistics(df)
    print("OK")

    # Створення цільових змінних
    print("6. Створення цільових змінних...", end=' ')
    df = create_target_variables(df)
    print("OK")

    # Валідація та очищення
    print("7. Валідація та очищення даних...")
    df = validate_and_clean_data(df)

    # Видалення перших рядків з нестабільними індикаторами
    print(f"8. Видалення перших {ROWS_TO_SKIP} рядків...", end=' ')
    df = df.iloc[ROWS_TO_SKIP:].reset_index(drop=True)
    print("OK")

    # Збереження результатів
    print("9. Збереження результатів...")
    save_processed_data(df, TECHNICAL_DATA_EURUSD_H1)

    # Статистика
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПІДГОТОВЛЕНИХ ДАНИХ:")
    print("=" * 60)
    print(f"Загальна кількість ознак: {len([col for col in df.columns if col not in ['Date']])}")
    print(f"Записів після обробки: {len(df)}")
    print(f"Період даних: {df['Date'].min()} - {df['Date'].max()}")

    # Приклад даних
    print("\nПерші 3 рядки технічних ознак:")
    feature_cols = [col for col in df.columns if
                    col not in ['Date'] and not col.startswith('Target') and not col.startswith('Direction')][:10]
    print(df[feature_cols].head(3).round(6))

    print("\nОстанні 3 рядки:")
    print(df[feature_cols].tail(3).round(6))

    print("\n" + "=" * 60)
    print("ПІДГОТОВКА ДАНИХ ЗАВЕРШЕНА УСПІШНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()