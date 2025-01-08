import pandas as pd

EURUSDM1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\EURUSDM1.csv"
TECHNICAL_DATA_EURUSDM1 = r"C:\Users\Oleksii\PycharmProjects\Druid\fundamental_data\technical_data_eurusd.csv"

# Завантажуємо вихідний CSV файл
df = pd.read_csv(EURUSDM1, encoding='utf-16', header=None)

# Задаємо назви колонок
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Other']

# Перетворюємо стовпець 'Date' на datetime формат
df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')

# Створюємо новий стовпець 'Local time' у форматі 'dd.mm.yyyy hh:mm:ss.000 GMT+0200'
df['Local time'] = df['Date'].dt.strftime('%d.%m.%Y %H:%M:%S.000 GMT+0200')

# Створюємо нову DataFrame з потрібними колонками
df_new = df[['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']]

data = pd.DataFrame

data = df[['Close']].copy()

# Add moving average, window = 10
data['MA10'] = df['Close'].rolling(10).mean()

# Add MACD
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
data['MACD'] = macd

# Add ROC, period = 2
n_steps = 2


def my_fun(x):
    return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]


data['ROC'] = data['Close'].rolling(n_steps).apply(my_fun)

# Add Momentum, period = 4
n_steps = 4


def my_fun(x):
    return (x.iloc[-1] - x.iloc[0])


data['Momentum'] = data['Close'].rolling(n_steps).apply(my_fun)

# Add RSI
# get the price diff
delta = data['Close'].diff()

# positive gains (up) and negative gains (down) Series
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0

period = 10

_gain = up.ewm(alpha=1.0 / period, adjust=True).mean()
_loss = down.abs().ewm(alpha=1.0 / period, adjust=True).mean()
RS = _gain / _loss

data["RSI"] = 100 - (100 / (1 + RS))

# Add Bollinger Bands , period = 20

typical_price = (df['Close'] + df['Low'] + df['High']) / 3
std = typical_price.rolling(20).std(ddof=0)
ma_tp = typical_price.rolling(20).mean()
data['BOLU'] = ma_tp + 2 * std
data['BOLD'] = ma_tp - 2 * std

# Add CCI Commodity channel index, period = 20
tp_rolling = typical_price.rolling(20)

# calculate mean deviation
mad = tp_rolling.apply(lambda s: abs(s - s.mean()).mean(), raw=True)

data["CCI"] = (typical_price - tp_rolling.mean()) / (0.015 * mad)

data.to_csv(TECHNICAL_DATA_EURUSDM1)
