import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta


def get_mt5_data(symbol, timeframe, start_date, end_date):
    """
    Завантажує історичні дані з MetaTrader 5.

    Args:
        symbol (str): Символ валютної пари (наприклад, "EURUSD").
        timeframe (int): Таймфрейм (наприклад, mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M15).
        start_date (datetime): Початкова дата.
        end_date (datetime): Кінцева дата.

    Returns:
        pd.DataFrame: Датафрейм з історичними даними.
    """
    if not mt5.initialize():
        print("MetaTrader 5 не вдалося ініціалізувати.")
        return None

    # Встановлюємо символ
    if not mt5.symbol_select(symbol, True):
        print(f"Символ {symbol} недоступний.")
        return None

    # Завантаження даних
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print("Дані не були завантажені.")
        return None

    # Конвертація даних у DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

if __name__ == "__main__":
    # Налаштування
    EURUSDM1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\EURUSD_M1.csv"
    SYMBOL = "EURUSD"  # Валютна пара
    TIMEFRAME = mt5.TIMEFRAME_M1  # Таймфрейм: M1

    # Кінцева дата — поточна дата
    END_DATE = datetime.now()
    # Початкова дата — 69 днів до кінцевої
    START_DATE = END_DATE - timedelta(days=69)

    # Завантаження даних
    data = get_mt5_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if data is not None:
        # Збереження у CSV
        data.to_csv(EURUSDM1, index=False)
        print(f"Дані збережено у файл: {EURUSDM1}")
        print("Перші два рядки:")
        print(data.head(2))
        print("\nОстанні два рядки:")
        print(data.tail(2))
    else:
        print("Не вдалося отримати дані.")

    # Завершення роботи MetaTrader 5
    mt5.shutdown()
