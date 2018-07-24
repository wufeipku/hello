import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import pymysql
from pprint import pprint
from func import coin_component,index_statistics

def component():
    # 计算每期成分股

    startdate = '2017-01-01'
    statistic_date = []
    while dt.datetime.strptime(startdate ,'%Y-%m-%d') < dt.datetime.today():
        statistic_date.append(startdate)
        startdate = (dt.datetime.strptime(startdate ,'%Y-%m-%d') + relativedelta(months = 3)).strftime('%Y-%m-%d')

    component_dict = {}
    for standard_date in statistic_date:
        component = list(coin_component(standard_date))
        if len(component) < 30:
            component.extend([''] * (30-len(component)))
        component_dict[standard_date] = component
    print(component_dict)
    component_pd = pd.DataFrame(component_dict)
    component_pd.to_csv('d:/habo/data/HAB30/component.csv')

   #计算权重
    #component_dict = dict(pd.read_csv('d:/habo/data/HAB30/component.csv'))
    date = []
    weight = {}
    engine = pymysql.connect(host='sh-cdb-s089fj1s.sql.tencentcdb.com', user='cur_read2', passwd='2tF6YSq45C43',
                             port=63405, db='currencies')
    for i in range(len(statistic_date)):
        date.append(dt.datetime.strptime(statistic_date[i], '%Y-%m-%d'))
        sql = "select coin_id, statistic_date, market_value_usd from coin_trading_copy where statistic_date = '{}' order by statistic_date " \
            .format(date[-1])
        df = pd.read_sql(sql, engine)
        df.fillna('ffill', inplace=True)
        list_marketvalue = []
        for coin_id in component_dict[statistic_date[i]]:
            if coin_id != '':
                market_value = df[df.coin_id == coin_id].iloc[0].market_value_usd
                list_marketvalue.append([coin_id, market_value, 1.])

        print(index_statistics(list_marketvalue)['ratio'].tolist())
        weight[statistic_date[i]] = index_statistics(list_marketvalue)['ratio'].tolist()
        print(weight[statistic_date[i]])
        if len(weight[statistic_date[i]]) < 30:
            weight[statistic_date[i]].extend([np.nan]*(30 - len(weight[statistic_date[i]])))

    weight = pd.DataFrame(weight)
    weight.to_csv('d:/habo/data/HAB30/weight.csv')

    pprint(weight)

if __name__ == '__main__':
    component()