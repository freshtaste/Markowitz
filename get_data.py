import numpy as np
import datetime as dt
import os
import pandas_datareader.data as web
import pandas as pd

        
def get_data_from_yahoo(start_year, end_year=None):
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
        
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    tickers = list(df['Symbol'])
    
    start = dt.datetime(start_year, 1, 1)
    if end_year:
        end = dt.datetime(end_year, 1, 1)
    else:
        end = dt.datetime.now()
        
    R = list()
    for ticker in tickers:
        try:
            df = web.DataReader(ticker, 'yahoo', start, end)
            p_close = df['Adj Close'].values
            R.append(p_close)
            print("{} Rechieved.".format(ticker))
        except:
            print("{} not available.".format(ticker))
        
    return np.array(R)

if __name__ == '__main__':
    start, end = 2018, None
    R_av = get_data_from_yahoo(start, end)
    np.save('data/s&p500-{}-{}.npy'.format(start, end), R_av)