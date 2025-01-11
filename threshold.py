import pandas as pd
import numpy as np
import math

# Функція для знаходження верхньої межі порогу
def get_upper_threshold(close):
    difference = close.diff().abs()  # Різниці між цінами
    bins = pd.cut(difference, bins=10)  # Розбивка на 10 інтервалів
    bins = bins.value_counts().to_frame().reset_index()
    bins["index"] = bins["index"].apply(lambda x: x.right)  # Оновлення меж
    bins = bins.to_numpy()  # Перетворення в масив
    percentile_count = len(difference) * 0.85  # 85% від загальної кількості
    count = 0
    for i in range(10):
        count += bins[i, 1]
        if count > percentile_count:
            return bins[i, 0]  # Повертає верхню межу порогу

# Функція для обчислення ентропії
def get_entropy(labels, base=None):
    vc = pd.Series(labels).value_counts(normalize=True)
    base = math.e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()  # Розрахунок ентропії

# Функція для знаходження найкращого порогу
def get_threshold(close):
    difference = close.diff().drop(0).tolist()  # Різниці між цінами
    thres_upper_bound = get_upper_threshold(close)  # Верхня межа порогу
    best_entropy = -float('inf')  # Початкове значення ентропії
    threshold = 0
    temp_thres = 0
    while temp_thres < thres_upper_bound:
        labels = [2 if diff > temp_thres else 1 if -diff > temp_thres else 0 for diff in difference]
        entropy = get_entropy(labels)  # Розрахунок ентропії
        if entropy > best_entropy:
            best_entropy = entropy
            threshold = temp_thres
        temp_thres += 0.00001  # Збільшення порогу
    return threshold  # Повертає найкращий поріг
