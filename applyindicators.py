import pandas as pd

# Шляхи до файлів
EURUSDM1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\EURUSDM1.csv"
TECHNICAL_DATA_EURUSDM1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\technical_data_eurusd.csv"

# Завантаження вихідного CSV файлу
df = pd.read_csv(EURUSDM1, encoding='utf-16', header=None)

# Встановлення назв колонок
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Other']

# Перетворення стовпця 'Date' у формат datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')

# Додавання стовпця 'Local time' у заданому форматі
df['Local time'] = df['Date'].dt.strftime('%d.%m.%Y %H:%M:%S.000 GMT+0200')

# Формування нового DataFrame з потрібними колонками
df_new = df[['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Копіювання лише стовпця 'Close' для подальших обчислень
data = df[['Close']].copy()

# Додавання ковзної середньої (Moving Average) з вікном 10
data['MA10'] = df['Close'].rolling(10).mean()

# Додавання MACD (Moving Average Convergence Divergence)
exp1 = data['Close'].ewm(span=12, adjust=False).mean()  # Швидка EMA
exp2 = data['Close'].ewm(span=26, adjust=False).mean()  # Повільна EMA
macd = exp1 - exp2  # Різниця між EMA
data['MACD'] = macd

# Додавання ROC (Rate of Change) з періодом 2
n_steps = 2


def my_fun(x):
    return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]


data['ROC'] = data['Close'].rolling(n_steps).apply(my_fun)

# Додавання Momentum з періодом 4
n_steps = 4


def my_fun(x):
    return (x.iloc[-1] - x.iloc[0])


data['Momentum'] = data['Close'].rolling(n_steps).apply(my_fun)

# Додавання RSI (Relative Strength Index)
delta = data['Close'].diff()  # Зміна ціни

# Виділення позитивних і негативних змін
up, down = delta.copy(), delta.copy()
up[up < 0] = 0  # Залишаємо лише зростання
down[down > 0] = 0  # Залишаємо лише спадання

# Розрахунок середнього приросту і втрати
period = 10
_gain = up.ewm(alpha=1.0 / period, adjust=True).mean()
_loss = down.abs().ewm(alpha=1.0 / period, adjust=True).mean()
RS = _gain / _loss

# Обчислення RSI
data["RSI"] = 100 - (100 / (1 + RS))

# Додавання верхньої та нижньої меж Bollinger Bands
typical_price = (df['Close'] + df['Low'] + df['High']) / 3  # Типова ціна
std = typical_price.rolling(20).std(ddof=0)  # Стандартне відхилення
ma_tp = typical_price.rolling(20).mean()  # Ковзна середня
data['BOLU'] = ma_tp + 2 * std  # Верхня межа
data['BOLD'] = ma_tp - 2 * std  # Нижня межа

# Додавання CCI (Commodity Channel Index)
tp_rolling = typical_price.rolling(20)

# Розрахунок середнього відхилення
mad = tp_rolling.apply(lambda s: abs(s - s.mean()).mean(), raw=True)

# Обчислення CCI
data["CCI"] = (typical_price - tp_rolling.mean()) / (0.015 * mad)

# Збереження оброблених даних у новий CSV файл
data.to_csv(TECHNICAL_DATA_EURUSDM1)
