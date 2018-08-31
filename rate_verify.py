import pymysql
import numpy as np
import decimal
from matplotlib import pyplot
import datetime
import pandas as pd
import numpy as np

def main():
    con=pymysql.connect(host='39.107.248.189',user='root',passwd='Digdig@I0',db='digdig_io',port=6666)
    cur=con.cursor(cursor=pymysql.cursors.DictCursor)
    sql = '''SELECT	* FROM qy_coin_data AS data
              INNER JOIN qy_coin_score AS score ON data.day = score.date AND data.coin_id = score.coin_id
              order by data.day 
           '''
    cur.execute(sql)
    data = cur.fetchall()
    close = []
    score = []
    date = []
    id = []
    marketcap = []

    for l in data:
        id.append(l['coin_id'])
        date.append(l['day'])
        score.append(float(l['fundamental_score']))
        close.append(float(l['close_p']))
        if l['market_cap'] == None:
            l['market_cap'] = np.nan
        marketcap.append(float(l['market_cap']))

    da = pd.DataFrame({'id':id,'date':date,'score':score,'price':close,'marketcap':marketcap})
    #da = da[da['marketcap'].isnull() == False]
    start_date = datetime.date(2018,6,1)
    end_date = datetime.date(2018,6,30)
    da1 = da[da['date'] == start_date].reset_index(drop = True)
    da2 = da[da['date'] == end_date].reset_index(drop = True)
    da1['price2']= np.nan
    for i in range(len(da1)):
        if da1.at[i,'id'] in da2['id'].values:
            da1.at[i,'price2'] = da2[da2['id'] == da1.loc[i,'id']].iat[0,3]

    da1['range'] = da1['price2']/da1['price']-1
    da1 = da1[da1['price2'].isnull() == False].reset_index(drop = True)
    #da1['range_01'] = da1['range'].apply(lambda x: 1 if x > 0 else 0)
    da1['score'] = da1['score'].apply(int)
    #s=da1.groupby(['score','range_01']).size()
    s = da1.groupby(['score']).mean()
    s[['range']].to_csv('d:\habo\data\score.csv')
    print(s['range'])
if __name__ == '__main__':
    main()
