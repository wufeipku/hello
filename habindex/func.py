import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import pymysql
from pprint import pprint
import numba
from matplotlib import pyplot as plt


def coin_component(statistic_date):
    """
    成分虚拟币
    :param statistic_date:
    :return component_coin:
    """
    # 整理日期
    statistic_date = dt.datetime.strptime(statistic_date, '%Y-%m-%d')
    date_stable = str(statistic_date - relativedelta(months=6))
    first_date = [statistic_date - relativedelta(months=i) for i in range(6, 0, -1)]
    last_date = [statistic_date - relativedelta(months=i) - relativedelta(days=1) for i in range(5, -1, -1)]
    date_zip = zip(first_date, last_date)
    engine = pymysql.connect(host = 'sh-cdb-s089fj1s.sql.tencentcdb.com',user='cur_read2',passwd='2tF6YSq45C43',port=63405,db='currencies')

    # 稳定交易六个月的虚拟币
    sql = "select distinct coin_id,ico_date from coin_info_copy"
    df = pd.read_sql(sql, engine)
    df = df[df.ico_date < date_stable]
    component_stable = set(df.coin_id.drop_duplicates().tolist())

    # 过去6个月市值连续超过3个月排名在100名以内
    ranks = []
    for date in date_zip:
        sql = "select coin_id,market_value_usd,turnover_usd from coin_trading_copy where statistic_date BETWEEN '{}' and '{}' order by statistic_date" \
            .format(date[0], date[1])
        df = pd.read_sql(sql, engine)
        coin_ids = df.coin_id.drop_duplicates().tolist()
        market_value_index = df.groupby('coin_id').sum().sort_values(by='market_value_usd',
                                                                     ascending=False).index.drop_duplicates().tolist()
        turnover_index = df.groupby('coin_id').sum().sort_values(by='turnover_usd',
                                                                 ascending=False).index.drop_duplicates().tolist()
        rank_dict = {}
        for coin_id in coin_ids:
            rank_dict[coin_id] = market_value_index.index(coin_id) + turnover_index.index(coin_id)
        rank = [i[0] for i in sorted(rank_dict.items(), key=lambda x: x[1])[:100]]
        ranks.append(set(rank))
    component_rank = (ranks[0] & ranks[1] & ranks[2]) | (ranks[1] & ranks[2] & ranks[3]) | (
            ranks[2] & ranks[3] & ranks[4]) | (ranks[3] & ranks[4] & ranks[5])

    # 分配token和coin的比例
    #使用我们的库获取分类
    sql = '''SELECT smyt_id,c.tags,cate,b.market_cap 
              FROM qy_coin_info a LEFT JOIN qy_coin_data b ON a.id = b.coin_id LEFT JOIN qy_coin_tags c ON a.id = c.coin_id
              WHERE b.day = '{}' 
          '''.format(statistic_date-dt.timedelta(days=1))
    engine = pymysql.connect(host = '60.205.223.152',user='op',passwd='op@123.',port=3306,db='digdig_io')
   # engine = pymysql.connect(host='39.107.248.189', user='root', passwd='Digdig@I0', port=6666, db='digdig_io')
    df_coin_standard = pd.read_sql(sql,engine)

    #df_coin_standard = pd.read_excel("d:/habo/app/currency/coin.xlsx", sheet_name="coin")
    df_coin_standard = df_coin_standard[df_coin_standard['tags'] != '锚定货币']
    market_value = df_coin_standard.groupby('cate').sum()
    token_market_value = market_value.loc[2, 'market_cap'] #tokens
    coin_market_value = market_value.loc[1, 'market_cap']  #coins
    token_number = int(round(30 * token_market_value / (token_market_value + coin_market_value)))
    coin_number = int(round(30 * coin_market_value / (token_market_value + coin_market_value)))

    # 缩减至30只成分币
    df_coin_standard.sort_values(by='market_cap', ascending=False, inplace=True)
    df_coin_standard.drop_duplicates(subset='smyt_id', keep='first', inplace=True)
   # coin_ids = df_coin_standard['smyt_id'].drop_duplicates().tolist()
    token_set = set(df_coin_standard[df_coin_standard['cate'] == 2]['smyt_id'].tolist())
    coin_set = set(df_coin_standard[df_coin_standard['cate'] == 1]['smyt_id'].tolist())
    token_sample = token_set & component_stable & component_rank
    coin_sample = coin_set & component_stable & component_rank

    print('component_rank:%d,token_sample %d,coin_sample %d'% (len(component_rank),len(token_sample),len(coin_sample)))

    #component_token = set()
    #component_coin = set()
    rank_token_total = []
    rank_coin_total = []
    engine = pymysql.connect(host='sh-cdb-s089fj1s.sql.tencentcdb.com', user='cur_read2', passwd='2tF6YSq45C43',
                             port=63405, db='currencies')
    date_zip = zip(first_date, last_date)
    for date in date_zip:
        sql = "select coin_id,market_value_usd,turnover_usd from coin_trading_copy where statistic_date BETWEEN '{}' and '{}' order by statistic_date"\
            .format(date[0], date[1])
        df = pd.read_sql(sql, engine)
        df_token = df[df['coin_id'].isin(token_sample)]
        df_coin = df[df['coin_id'].isin(coin_sample)]
        token_market_value_index = df_token.groupby('coin_id').sum().sort_values(by='market_value_usd',
                                                                     ascending=False).index.drop_duplicates().tolist()
        token_turnover_index = df_token.groupby('coin_id').sum().sort_values(by='turnover_usd',
                                                                                 ascending=False).index.drop_duplicates().tolist()
        coin_market_value_index = df_coin.groupby('coin_id').sum().sort_values(by='market_value_usd',
                                                                                 ascending=False).index.drop_duplicates().tolist()
        coin_turnover_index = df_coin.groupby('coin_id').sum().sort_values(by='turnover_usd',
                                                                 ascending=False).index.drop_duplicates().tolist()
        rank_token = {}
        rank_coin = {}
        for coin_id in token_sample:
            try:
                rank_token[coin_id] = token_market_value_index.index(coin_id) + token_turnover_index.index(coin_id)
            except:
                print('data missing: %s'% (coin_id))
        for coin_id in coin_sample:
            try:
                rank_coin[coin_id] = coin_market_value_index.index(coin_id) + coin_turnover_index.index(coin_id)
            except:
                print('data missing: %s' % (coin_id))

        rank_token_total.append(rank_token)
        rank_coin_total.append(rank_coin)
    rank_token_total = pd.DataFrame(rank_token_total).mean()
    rank_coin_total = pd.DataFrame(rank_coin_total).mean()
    rank_token_total = rank_token_total.sort_values(ascending=True).index.tolist()
    rank_coin_total = rank_coin_total.sort_values(ascending=True).index.tolist()
    if len(rank_token_total) < token_number:
        component_token = set(rank_token_total)
        component_coin = set(rank_coin_total[:coin_number+token_number-len(rank_token_total)])
    else:
        if len(rank_coin_total) < coin_number:
            component_coin = set(rank_coin_total)
            component_token = set(rank_token_total[:token_number+coin_number-len(rank_coin_total)])
        else:
            component_token = set(rank_token_total[:token_number])
            component_coin = set(rank_coin_total[:coin_number])
    component_coin = component_token | component_coin

    return component_coin

@numba.jit
def index_statistics(market_value_list):
    """
    成分虚拟币的市值权重
    :param market_value_list:
    :return df_statistics:
    """
    df_statistics = pd.DataFrame(market_value_list, columns=['coin_id', 'market_value', 'ratio'])
    df_statistics.index = df_statistics.coin_id
    del df_statistics['coin_id']

    df_statistics['adjust_market_value'] = df_statistics.market_value * df_statistics.ratio
    a = 0.25
    while max(df_statistics.adjust_market_value) / sum(df_statistics.adjust_market_value) > a:
       # df_statistics.sort_values(by='adjust_market_value', ascending=False, inplace=True)
        for coin_id in df_statistics.index.tolist():
            weight = (df_statistics.loc[coin_id, 'adjust_market_value'] / sum(df_statistics.adjust_market_value))
            if weight > a:
                df_statistics.loc[coin_id, 'ratio'] = df_statistics.loc[coin_id, 'ratio'] - 0.001
                #df_statistics.loc[coin_id, 'ratio'] =  round(a*(df_statistics.adjust_market_value.sum() - \
                 #                                      df_statistics.loc[coin_id,'adjust_market_value'])/\
                          #                                   (1-a)/df_statistics.loc[coin_id,'market_value'],4)

        df_statistics.adjust_market_value = df_statistics.market_value * df_statistics.ratio

    return df_statistics

@numba.jit()
def all_coin_index():
    """
    虚拟币指数
    :param:
    :return indicators:
    """
    indicators = []
    engine = pymysql.connect(host = 'sh-cdb-s089fj1s.sql.tencentcdb.com',user='cur_read2',passwd='2tF6YSq45C43',port=63405,db='currencies')
    sql = "select coin_id, statistic_date, market_value_usd from coin_trading_copy where statistic_date BETWEEN '{}' and '{}' order by statistic_date" \
        .format(dt.datetime.strptime('2017-01-01', '%Y-%m-%d'), dt.datetime.today())
    df = pd.read_sql(sql, engine)
    df.fillna('ffill', inplace=True)

    component_dict = dict(pd.read_csv('d:/habo/data/HAB30/component.csv'))
    weight_dict = dict(pd.read_csv('d:/habo/data/HAB30/weight.csv'))
    # 基期市值
    list_base = []
    component = component_dict['2017-01-01']
    for coin_id in component:
        if coin_id is not np.nan:
            market_value = df[df.coin_id == coin_id].iloc[0].market_value_usd
            list_base.append([coin_id, market_value, 1.])

    market_value_base = sum(index_statistics(list_base).adjust_market_value)

    # 每日市值
    market_date_list = list(pd.date_range(start='2017-01-01', end=dt.datetime.today() - dt.timedelta(1)))
    for i in range(len(market_date_list)):
        statistic_date = market_date_list[i].strftime("%Y-%m-%d")
        if int((market_date_list[i].month - 1) / 3) * 3 + 1 < 10:
            component_date = str(market_date_list[i].year) + '-0' + str(
                int((market_date_list[i].month - 1) / 3) * 3 + 1) + '-01'
        else:
            component_date = str(market_date_list[i].year) + '-' + str(
                int((market_date_list[i].month - 1) / 3) * 3 + 1) + '-01'
        component = component_dict[component_date]
        weight = weight_dict[component_date]
        #list_daily = []
        adjust_market_value = []
        for j in range(len(component)):
            if component[j] is not np.nan:
                df_coin = df[df.coin_id == component[j]]
                df_coin.index = pd.to_datetime(df_coin.statistic_date)
                df_coin = df_coin.reindex(index=market_date_list, method='ffill')
                adjust_market_value.append(df_coin.loc[statistic_date, 'market_value_usd'] * weight[j])
        #    list_daily.append([coin_id, market_value, 1.])

        market_value_daily = sum(adjust_market_value)

        # 虚拟币指数

        coin_index_value = round(market_value_daily * 100 / market_value_base, 4)
        #基期基数调整
        if statistic_date[5:10] in ['12-31','03-31','06-30','09-30']:
            nextdate = (dt.datetime.strptime(statistic_date,"%Y-%m-%d") + relativedelta(days = 1)).strftime('%Y-%m-%d')
            component = component_dict[nextdate]
            weight = weight_dict[nextdate]
            adjust_market_value_base = []
            for j in range(len(component)):
                if component[j] is not np.nan:
                    df_coin = df[df.coin_id == component[j]]
                    df_coin.index = pd.to_datetime(df_coin.statistic_date)
                    df_coin = df_coin.reindex(index=market_date_list, method='ffill')
                    adjust_market_value_base.append(df_coin.loc[statistic_date, 'market_value_usd'] * weight[j])

            market_value_new = sum(adjust_market_value_base)
            market_value_base = market_value_base * market_value_new / market_value_daily

        indicators.append({"statistic_date": statistic_date, 'value': coin_index_value})
        pprint(indicators[-1])

    return indicators

#指数回测，以BTC为基准
def back_test(indicators):

    engine = pymysql.connect(host = 'sh-cdb-s089fj1s.sql.tencentcdb.com',user='cur_read2',passwd='2tF6YSq45C43',port=63405,db='currencies')
    sql = "SELECT statistic_date,current_price_usd as price FROM coin_trading_copy WHERE coin_id = 'JRB100001' and statistic_date between '{}' and '{}' order by statistic_date".\
        format(dt.datetime.strptime(indicators['statistic_date'][0],'%Y-%m-%d'),dt.datetime.strptime(indicators['statistic_date'][len(indicators)-1],'%Y-%m-%d'))
    btc = pd.read_sql(sql,engine)
    hab_index = indicators
    btc.index = btc.statistic_date
    btc = btc.reindex(index = pd.to_datetime(hab_index['statistic_date']).tolist(),method = 'ffill')
    btc['profit'] = btc['price']/btc.iloc[0,1] - 1
    hab_index['profit'] = hab_index['value']/hab_index.loc[0,'value'] - 1
    #绘制收益曲线
    figuresize = 11,9
    figure= plt.figure(figsize=figuresize)
    plt.title('HAB_index acc_profit')
    plt.plot(btc['statistic_date'],hab_index['profit'],color = 'red',label = 'HAB30')
    plt.plot(btc['statistic_date'],btc['profit'],color = 'blue',label = 'BTC')
    plt.legend()
    plt.tick_params(labelsize=12)
    plt.ylabel('Acc_profit')
    plt.savefig('d:/habo/data/HAB30/HAB_BTC_a0.25.png')
    plt.show()

    hab = hab_index['value'].tolist()
    #夏普率
    #年化收益
    index_profit_a = (hab[-1]/hab[0])**(365/len(hab)) - 1
    #标准差
    index_profit = (hab_index['value'] / hab_index['value'].shift(1) - 1).tolist()
    index_profit[0] = 0
    index_profit_avg = np.mean(index_profit)
    index_std_a = (sum([(index_profit[i]-index_profit_avg)**2 for i in range(len(index_profit))])/(len(index_profit)-1))**0.5 * 365 ** 0.5
    sharpe = index_profit_a / index_std_a
    print('年化收益：',index_profit_a)
    print('夏普率：',sharpe)

    #最大回撤
    backtrace = []
    for i in range(len(hab)):
        temp = hab[i]/max(hab[:i+1]) - 1
        if temp > 0:
            temp  = 0
        backtrace.append(temp)
    MBT = min(backtrace)
    print('最大回撤：',MBT)

    #累计收益
    hab_acc_profit = hab[-1]/hab[0] - 1
    print('HAB累计收益',hab_acc_profit)
    #基准收益
    btc_acc_profit = btc['profit'][len(btc)-1]
    print('BTC累计收益：',btc_acc_profit)