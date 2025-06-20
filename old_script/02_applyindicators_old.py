import pandas as pd
import sys
from pathlib import Path

# Константи
EURUSD_H1 = Path(r"/historic_data/csv/EURUSDH1.csv")
TECHNICAL_DATA_EURUSD_H1 = Path(r"/historic_data/csv/historic_data_eurusd_h1.csv")
ROWS_TO_SKIP = 30  # Кількість рядків для видалення

# Параметри технічних індикаторів
BOLLINGER_PERIOD = 20
RSI_PERIOD = 10
MACD_FAST = 12
MACD_SLOW = 26
ROC_PERIOD = 2
MOMENTUM_PERIOD = 4
CCI_PERIOD = 20


def load_data(file_path: Path) -> pd.DataFrame:
    """Завантаження та початкова обробка даних."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Помилка: Файл {file_path} не знайдено")
        sys.exit(1)
    except Exception as e:
        print(f"Помилка при завантаженні даних: {str(e)}")
        sys.exit(1)


def calculate_roc(x: pd.Series) -> float:
    """Розрахунок Rate of Change."""
    return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]


def calculate_momentum(x: pd.Series) -> float:
    """Розрахунок Momentum."""
    return x.iloc[-1] - x.iloc[0]


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Додавання всіх технічних індикаторів."""
    # Копіювання необхідних стовпців
    result = data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()

    # Moving Average
    result['MA10'] = data['Close'].rolling(10).mean()

    # MACD
    exp1 = data['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = data['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    result['MACD'] = exp1 - exp2

    # ROC
    result['ROC'] = data['Close'].rolling(ROC_PERIOD).apply(calculate_roc)

    # Momentum
    result['Momentum'] = data['Close'].rolling(MOMENTUM_PERIOD).apply(calculate_momentum)

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1.0 / RSI_PERIOD, adjust=True).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1.0 / RSI_PERIOD, adjust=True).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    typical_price = (data['Close'] + data['Low'] + data['High']) / 3
    std = typical_price.rolling(BOLLINGER_PERIOD).std(ddof=0)
    ma_tp = typical_price.rolling(BOLLINGER_PERIOD).mean()
    result['BOLU'] = ma_tp + 2 * std
    result['BOLD'] = ma_tp - 2 * std

    # CCI
    tp_rolling = typical_price.rolling(CCI_PERIOD)
    mad = tp_rolling.apply(lambda s: abs(s - s.mean()).mean(), raw=True)
    result['CCI'] = (typical_price - tp_rolling.mean()) / (0.015 * mad)

    # Видалення перших N рядків
    result = result.iloc[ROWS_TO_SKIP:]

    return result


def validate_data(data: pd.DataFrame) -> None:
    """Перевірка якості даних."""
    if data.isna().any().any():
        print("\nУвага: Виявлено пропущені значення в наступних стовпцях:")
        print(data.isna().sum()[data.isna().sum() > 0])

    if (data.isin([float('inf'), float('-inf')])).any().any():
        print("\nУвага: Виявлено нескінченні значення в даних")


def save_data(data: pd.DataFrame, file_path: Path) -> None:
    """Збереження даних у файл."""
    try:
        data.to_csv(file_path, index=False)
        print(f"\nДані успішно збережено у файл: {file_path}")
    except Exception as e:
        print(f"Помилка при збереженні даних: {str(e)}")
        sys.exit(1)


def main():
    # Завантаження даних
    print("Завантаження даних...", end=' ')
    df = load_data(EURUSD_H1)
    print("OK")

    # Розрахунок індикаторів
    print("Розрахунок технічних індикаторів...", end=' ')
    technical_data = add_indicators(df)
    print("OK")

    # Валідація даних
    print("Перевірка якості даних...", end=' ')
    validate_data(technical_data)
    print("OK")

    # Збереження результатів
    save_data(technical_data, TECHNICAL_DATA_EURUSD_H1)

    # Виведення прикладу даних
    print("\nПерші два рядки:")
    print(technical_data.head(2))
    print("\nОстанні два рядки:")
    print(technical_data.tail(2))


if __name__ == "__main__":
    main()