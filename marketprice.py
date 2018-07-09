import tushare as ts
import pandas as pd
import datetime
import numpy as np
def main():
    code = []
    time = []
    profit1 = []
    profit2 = []
    df = ts.forecast_data(2016, 2)
    for i in range(len(df)):
        if df['pre_eps'][i] > 0.1:
            code.append(df['code'][i])
            time.append(df['report_date'][i])

    for i in range(len(code)):
        start_time=datetime.datetime.strptime(time[i],'%Y-%m-%d')
        end_time1=start_time+datetime.timedelta(days=30)
        end_time2=start_time+datetime.timedelta(days=60)
        endstr1=end_time1.strftime('%Y-%m-%d')
        endstr2=end_time2.strftime('%Y-%m-%d')
        data = ts.get_k_data(start=time[i], end=endstr1, ktype='D', autype='qfq', code=code[i])
        try:
            profit1.append(data['close'][data.index[-1]]/data['close'][data.index[0]]-1)
        except:
            profit1.append(np.nan)
        data = ts.get_k_data(start=time[i], end=endstr2, ktype='D', autype='qfq', code=code[i])
        try:
            profit2.append(data['close'][data.index[-1]] / data['close'][data.index[0]]-1)
        except:
            profit2.append(np.nan)
    output={
        'code':code,
        'profit1':profit1,
        'profit2': profit2
    }
    a=pd.DataFrame(output)
    a.to_csv('D:/pre_profit.csv')

if __name__ == '__main__':
    main()
