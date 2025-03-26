import requests
import pandas as pd
import gzip
from io import BytesIO

# URL для скачування даних у форматі TSV
url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/prc_hicp_manr?format=TSV&compressed=true"

# Отримуємо дані
response = requests.get(url)

# Перевіряємо статус відповіді
if response.status_code == 200:
    # Відкриваємо стиснутий файл
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
        # Читаємо TSV дані
        df = pd.read_csv(f, sep="\t", header=0)

    # Виводимо перші кілька рядків DataFrame
    print(df)
else:
    print(f"Помилка завантаження: {response.status_code}")
