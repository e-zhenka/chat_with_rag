import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from config import Settings
import requests

settings = Settings.from_yaml("config.yaml")


def get_stock_info(ticker_symbol):
    current_date = datetime.now().strftime('%Y-%m-%d')
    two_months_ago = (datetime.now() - relativedelta(months=3)).strftime('%Y-%m-%d')

    # todo в settings
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker_symbol}/candles.json?from={two_months_ago}&till={current_date}Ъ&interval=24"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["candles"]["data"], columns=data["candles"]["columns"])

    df['time'] = pd.to_datetime(df['begin'])
    df = df.set_index('begin')

    # Определение цвета в зависимости от изменения цены
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    color = '#FF3333' if end_price < start_price else '#00CC66'  # Красный или зеленый

    # Создание графика
    plt.style.use('dark_background')  # Темный фон для современного вида
    fig, ax = plt.subplots(figsize=(12, 6))

    # Отрисовка линии
    ax.plot(df.index, df['close'], color=color, linewidth=2)
    ax.scatter(df.index[-1], end_price, color='white', s=60, zorder=5)
    ax.scatter(df.index[-1], end_price, color=color, s=40, zorder=5)

    ymin = df['close'].min() * 0.95  # Нижняя граница чуть ниже минимальной цены

    # Заполнение области под графиком с градиентом
    ax.fill_between(df.index, df['close'], ymin,
                    facecolor=color,
                    alpha=0.3)  # Базовая прозрачность

    # Настройка осей
    ax.set_facecolor('#1A1A1A')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#404040')
    ax.spines['bottom'].set_color('#404040')

    # Линии разметки
    ax.tick_params(axis='x', colors='#808080', rotation=45)
    ax.tick_params(axis='y', colors='#808080')

    # Заголовок и подписи
    plt.title(f'{ticker_symbol} 1D',
              color='#808080',
              fontsize=14,
              pad=20)
    ax.set_ylabel('Price (RUB)', color='#808080')

    # Оптимизация компоновки
    plt.tight_layout()

    # Вертикальные линии
    ax.set_xticks([])
    custom_x = df[(df.time.dt.weekday == 0)]['time'].dt.strftime('%d.%m')
    for label, x_pos in zip(custom_x, custom_x.index):
        ax.axvline(x=x_pos, color='#808080', linestyle='--', linewidth=1, alpha=0.3)
        ax.text(x_pos, ymin * 0.95, label, color='#808080', ha='center', va='bottom', rotation=45)

    filename = settings.img_path
    plt.savefig(filename, format='png')
    plt.close()

    difference = (end_price / start_price - 1) * 100 if end_price > start_price else (start_price / end_price - 1) * 100

    context = [
        f'Текущая цена: {round(end_price, 2)} руб.',
        F'За последние 2 месяца акция {"выросла" if end_price > start_price else "упала"} на {round(difference, 2)} %'
    ]

    return context


# if __name__ == '__main__':
#     e = get_stock_info('SBER')
#     pass
