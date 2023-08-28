import warnings
warnings.filterwarnings('ignore')

from time import time
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

# 함수 정의

def chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

###

idx = pd.IndexSlice
results_path = Path('KR2_results', 'asset_pricing') # 경로 설정

if not results_path.exists():
    results_path.mkdir(parents=True)

krx = fdr.StockListing('KRX')
krx = krx[krx["Market"] != "KONEX"]
krx = krx[krx["Market"] != "KOSDAQ"]
krx['Code'] = krx['Code'] + '.' + krx['Market'].apply(lambda x: 'KS')
krx = krx["Code"].to_list()
n = len(krx)

yf_codes = yf.Tickers(krx)

meta_data = []
start = time()
for code in tqdm(krx):
    try:
        yf_object = yf.Ticker(code)
        s = pd.Series(yf_object.get_info())
        meta_data.append(s.to_frame(code))
    except Exception as e:
        print(code, e)

df = pd.concat(meta_data, axis=1).dropna(how='all').T
df = df.apply(pd.to_numeric, errors='ignore')

# 다운로드 받을 주식 데이터의 메타 정보 저장
df.to_hdf(results_path / 'data.h5', 'stocks/info')

prices_adj = []
start = time()
for i, chunk in enumerate(chunks(krx, 100), 1):
    prices_adj.append(yf.download(chunk, period='max', auto_adjust=True).stack(-1))

    per_ticker = (time()-start) / (i * 100)
    to_do = n - (i * 100)
    to_go = to_do * per_ticker    
    print(f'Success: {len(prices_adj):5,.0f}/{i:5,.0f} | To go: {format_time(to_go)} ({to_do:5,.0f})')

prices_adj = (pd.concat(prices_adj)
              .dropna(how='all', axis=1)
              .rename(columns=str.lower)
              .swaplevel())

prices_adj.index.names = ['ticker', 'date']

# 이상치 제거
df = prices_adj.close.unstack('ticker')
pmax = df.pct_change().max()
pmin = df.pct_change().min()
to_drop = pmax[pmax > 1].index.union(pmin[pmin<-1].index)
print("Outliers :", len(to_drop))

prices_adj = prices_adj.drop(to_drop, level='ticker')
print("Final using Stock Data :", len(prices_adj.index.unique('ticker')))

# 최종 데이터셋 저장
prices_adj.sort_index().loc[idx[:, '1993': '2023'], :].to_hdf(results_path / 'data.h5', 
                                                              'stocks/prices/adjusted')