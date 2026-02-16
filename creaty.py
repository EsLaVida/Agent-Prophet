import pandas as pd
import numpy as np

def create_dummy_csv():
    dates = pd.date_range(start='2024-01-01', end='2025-02-01', freq='D')
    
    # Продажи с недельным трендом
    sales = 100 + np.arange(len(dates)) * 0.5 + np.sin(np.arange(len(dates)) * (2 * np.pi / 7)) * 20
    pd.DataFrame({'ds': dates, 'y': sales}).to_csv('sales.csv', index=False)
    
    # Поездки с сильной сезонностью
    trips = 50 + np.random.normal(0, 5, len(dates)) + (dates.dayofweek < 5) * 30
    pd.DataFrame({'ds': dates, 'y': trips}).to_csv('trips.csv', index=False)
    
    # Цена (плавающий тренд)
    price = 1500 + np.cumsum(np.random.normal(0, 2, len(dates)))
    pd.DataFrame({'ds': dates, 'y': price}).to_csv('price.csv', index=False)

create_dummy_csv()