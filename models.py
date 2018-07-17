import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
from utils.db_config import engine_currency
from pprint import pprint
from pymongo import MongoClient
import numba


def coin_index_mongo():
    """
    虚拟币指数缓存数据
    :param:
    :return coin_index:
    """
    client = MongoClient('111.231.84.86', 9080)
    db = client['fof_api_currency']
    data = {'datetime': str(dt.date.today())}

    if db.coin_index.find_one(data):
        resp_data = db.coin_index.find_one(data)
        del resp_data['_id']
        del resp_data['datetime']

    else:
        indicators = all_coin_index(update_date='2017-1-1')
        resp_data = {
            'coin_index': indicators,
        }
        db.coin_index.insert_one({**resp_data, **data})

    return resp_data


def date_range_convert(coin_id, freq_length):
    """
    freq_length转date_range
    :param coin_id:
    :param freq_length:
    :return date_range, interval:
    """
    engine = engine_currency()

    date_range = {'min': None, 'max': None}
    interval = {'min': None, 'max': None}

    if coin_id:
        sql = "select min(statistic_date) as min_statistic_date, max(statistic_date) as max_statistic_date from coin_trading where coin_id='{}'".format(
            coin_id)
    else:
        sql = "select min(statistic_date) as min_statistic_date, max(statistic_date) as max_statistic_date from coin_trading"
    df = pd.read_sql(sql, engine)

    if len(df):
        date_start = df.loc[0, 'min_statistic_date']
        date_end = df.loc[0, 'max_statistic_date']
        interval['min'] = str(date_start)
        interval['max'] = str(date_end)

        if freq_length == 'total':
            date_range['min'] = str(date_start)
            date_range['max'] = str(date_end)
        elif freq_length in ['m1', 'm3', 'm6', 'y1', 'y2', 'y3', 'y5']:
            _freq_map = {'m1': 1, 'm3': 3, 'm6': 6, 'y1': 12, 'y2': 24, 'y3': 36, 'y5': 60}
            months = _freq_map.get(freq_length)

            date_start_new = date_end - relativedelta(months=months)
            if date_start_new >= date_start:
                date_range['min'] = str(date_start_new)
                date_range['max'] = str(date_end)
        elif freq_length == 'year':
            date_range['min'] = str(dt.date(date_end.year, 1, 1))
            date_range['max'] = str(date_end)
        return (date_range, interval)


def _get_prices_list_by_coin_id(coin_id, date_range):
    """
    根据coin_id获取价格序列
    """
    engine = engine_currency()
    if coin_id:
        sql = "select coin_id, statistic_date, current_price_usd, market_value_usd, turnover_usd from coin_trading where coin_id='{}' and statistic_date BETWEEN '{}' and '{}' order by coin_id, statistic_date" \
            .format(coin_id, date_range['min'], date_range['max'])
    else:
        sql = "select coin_id, statistic_date, current_price_usd, market_value_usd, turnover_usd from coin_trading where statistic_date BETWEEN '{}' and '{}' order by coin_id, statistic_date" \
            .format(date_range['min'], date_range['max'])

    df = pd.read_sql(sql, engine)
    return df


def coin_info(coin_name):
    """
    基本信息
    :param coin_name:
    :return coin_info:
    """
    coin_info = {}
    engine = engine_currency()
    if coin_name:
        sql = "select * from coin_info where LOCATE('{}', `english_name`)>0".format(coin_name)
        df = pd.read_sql(sql, engine)
        df = df[:].astype(str)
        if len(df):
            coin_info = df.loc[0].to_dict()
            coin_info['statistic_date'] = str(coin_info['statistic_date'])
        else:
            coin_info['msg'] = '没有找到相关币信息'
    else:
        sql = "select * from coin_info"
        df = pd.read_sql(sql, engine)
        df = df[:].astype('str')
        df = df.replace('－', "None")
        del df['coin_profile']
        coin_info = df.to_dict(orient='record')

    return coin_info


def coin_concept(statistic_date):
    """
    概念行情
    :param statistic_date:
    :return statistic_date,concepts:
    """
    engine = engine_currency()
    if statistic_date:
        sql = "select * from coin_concept where statistic_date='{}' order by statistic_date".format(statistic_date)
    else:
        sql = "select * from coin_concept order by statistic_date"
    df = pd.read_sql(sql, engine)

    print(df)

    if len(df):
        statistic_date = str(df.iloc[-1]['statistic_date'])
        del df['statistic_date']
        del df['update_time']
        concepts = df.to_dict(orient='records')

        return statistic_date, concepts
    else:
        return None, []


def indictors_return(coin_id, freq_length):
    """
    收益指标计算
    :param coin_id:
    :param freq_length:
    :return indicators:
    """
    indicators = []
    date_range, interval = date_range_convert(coin_id, freq_length)
    if date_range.get('min') is None or date_range.get('max') is None:
        return indicators

    df = _get_prices_list_by_coin_id(coin_id, date_range)
    df.fillna('ffill', inplace=True)
    if len(df) == 0:
        return indicators

    coin_ids = df.coin_id.drop_duplicates().tolist()
    all_coin_index = coin_index_mongo()['coin_index']

    for coin_id in coin_ids:

        df_coin = df[df.coin_id == coin_id]
        prices = df_coin.current_price_usd.tolist()
        statistic_dates = np.array(df_coin.statistic_date.tolist()).astype('str')
        df_coin_index = pd.DataFrame(all_coin_index)

        indicator = {
            'coin_id': coin_id,
            'return': [],
            'return_a': [],
            'return_q': [],
            'coin_index_return': []
        }

        for i in range(1, len(statistic_dates)):

            # 累计收益率
            daily_return = prices[i] / prices[0] - 1
            indicator['return'].append({
                "statistic_date": statistic_dates[i],
                'value': daily_return if daily_return or daily_return == 0 else None
            })

            # 年化收益率（控制最大年化收益不超过10000，防止内存溢出）
            if prices[i] > 10001 ** (i / 365) * prices[0]:
                daily_return_a = 10000
            else:
                daily_return_a = min((prices[i] / prices[0]) ** (365.0 / i) - 1, 10000)
            indicator['return_a'].append({
                "statistic_date": statistic_dates[i],
                'value': daily_return_a if daily_return_a or daily_return_a == 0 else None
            })

            # 指数收益率(2017-1-1前无值)
            try:
                base_coin_index = df_coin_index[df_coin_index.statistic_date == statistic_dates[0]].value.values[0]
                coin_index = df_coin_index[df_coin_index.statistic_date == statistic_dates[i]].value.values[0]
                coin_index_return = coin_index / base_coin_index - 1
                indicator['coin_index_return'].append({
                    "statistic_date": statistic_dates[i],
                    'value': coin_index_return if coin_index_return or coin_index_return == 0 else None
                })
            except:
                pass

        # 季度收益率
        df_q = df_coin.copy()
        df_q.index = pd.to_datetime(df_q.statistic_date)
        df_q_price = df_q.resample('Q').last().current_price_usd
        statistic_dates_q = df_q.resample('Q').last().index.tolist()
        statistic_dates_q[-1] = dt.datetime.strptime(statistic_dates[-1].astype('O'), "%Y-%m-%d")
        df_q_return = (df_q_price / df_q_price.shift(1) - 1).tolist()
        df_q_return[0] = 0
        for i in range(len(statistic_dates_q)):
            indicator['return_q'].append({
                "statistic_date": statistic_dates_q[i].strftime("%Y-%m-%d"),
                'value': df_q_return[i] if df_q_return[i] or df_q_return[i] == 0 else None
            })

        indicators.append(indicator)

    return indicators


@numba.jit
def indictors_risk(coin_id, freq_length):
    """
    风险指标计算
    :param coin_id:
    :param freq_length:
    :return indicators:
    """
    indicators = []

    date_range, interval = date_range_convert(coin_id, freq_length)
    if date_range.get('min') is None or date_range.get('max') is None:
        return indicators

    df = _get_prices_list_by_coin_id(coin_id, date_range)
    df.fillna('ffill', inplace=True)
    if len(df) == 0:
        return indicators

    coin_ids = df.coin_id.drop_duplicates().tolist()

    for coin_id in coin_ids:

        df_coin = df[df.coin_id == coin_id]
        prices = df_coin.current_price_usd.tolist()
        statistic_dates = np.array(df_coin.statistic_date.tolist()).astype('str')

        indicator = {
            'coin_id': coin_id,
            'max_retracement': [],
            'dynamic_retracement': [],
            'stdev': [],
            'annual_stdev': []
        }

        # 最大回撤/动态回撤
        MDD_list = []
        for i in range(1, len(statistic_dates)):
            price = prices[:i + 1]
            retracement = 1 - price[-1] / max(price[:-1])
            if retracement >= 0:
                DDD = retracement
                MDD_list.append(DDD)
            else:
                DDD = 0
            MDD = max(MDD_list)
            indicator['max_retracement'].append({
                "statistic_date": statistic_dates[i],
                'value': MDD if MDD or MDD == 0 else None
            })
            indicator['dynamic_retracement'].append({
                "statistic_date": statistic_dates[i],
                'value': DDD if DDD or DDD == 0 else None
            })

            # 标准差/年化标准差
            r_avg = sum(price) / len(price)
            list = []
            for x in price:
                list.append((x - r_avg) ** 2)
            s = (sum(list) / i) ** 0.5
            s_a = s * 365 ** 0.5
            indicator['stdev'].append({
                "statistic_date": statistic_dates[i],
                'value': s if s or s == 0 else None
            })
            indicator['annual_stdev'].append({
                "statistic_date": statistic_dates[i],
                'value': s_a if s_a or s_a == 0 else None
            })

        indicators.append(indicator)

    return indicators


def indictors_risk_adjust(coin_id, freq_length):
    """
    风险调整收益指标计算
    :param coin_id:
    :param freq_length:
    :return indicators:
    """
    indicators = []

    date_range, interval = date_range_convert(coin_id, freq_length)
    if date_range.get('min') is None or date_range.get('max') is None:
        return indicators

    df = _get_prices_list_by_coin_id(coin_id, date_range)
    df.fillna('ffill', inplace=True)
    if len(df) == 0:
        return indicators

    coin_ids = df.coin_id.drop_duplicates().tolist()

    for coin_id in coin_ids:

        df_coin = df[df.coin_id == coin_id]
        prices = df_coin.current_price_usd.tolist()
        statistic_dates = np.array(df_coin.statistic_date.tolist()).astype('str')

        indicator = {
            'coin_id': coin_id,
            'sharpe_a': []
        }

        for i in range(1, len(statistic_dates)):

            price = prices[:i + 1]

            # 年化收益率（控制最大年化收益不超过10000，防止内存溢出）
            if prices[i] > 10001 ** (i / 365) * prices[0]:
                daily_return_a = 10000
            else:
                daily_return_a = min((prices[i] / prices[0]) ** (365.0 / i) - 1, 10000)

            # 年化标准差
            r_avg = sum(price) / len(price)
            s = (sum([(x - r_avg) ** 2 for x in price]) / i) ** 0.5
            s_a = s * 365 ** 0.5

            # 年化夏普比
            indicator['sharpe_a'].append({
                "statistic_date": statistic_dates[i],
                'value': daily_return_a / s_a if s_a or s_a == 0 else None
            })

        indicators.append(indicator)

    return indicators


@numba.jit
def indictors_style(coin_id, freq_length):
    """
    风格指标计算
    :param coin_id:
    :param freq_length:
    :return indicators:
    """
    indicators = []

    date_range, interval = date_range_convert(coin_id, freq_length)
    if date_range.get('min') is None or date_range.get('max') is None:
        return indicators

    df = _get_prices_list_by_coin_id(coin_id, date_range)
    df.fillna('ffill', inplace=True)
    if len(df) == 0:
        return indicators

    coin_ids = df.coin_id.drop_duplicates().tolist()

    for coin_id in coin_ids:
        df_coin = df[df.coin_id == coin_id]
        statistic_dates = np.array(df_coin.statistic_date.tolist()).astype('str')
        market_values = df_coin.market_value_usd.tolist()
        turnovers = df_coin.turnover_usd.tolist()

        indicator = {
            'coin_id': coin_id,
            'daily_average_volumn': [],
            'daily_average_activity': []
        }

        for i in range(1, len(statistic_dates)):
            market_value = market_values[:i + 1]
            turnover = turnovers[:i + 1]

            # 日均交易量
            average_volumn = sum(turnover) / len(turnover)
            indicator['daily_average_volumn'].append({
                "statistic_date": statistic_dates[i],
                'value': average_volumn
            })

            # 日均市值
            average_market = sum(market_value) / len(market_value)
            average_activity = average_volumn / average_market

            # 日均活跃度
            indicator['daily_average_activity'].append({
                "statistic_date": statistic_dates[i],
                'value': average_activity if average_activity or average_activity == 0 else None
            })

        indicators.append(indicator)

    return indicators


@numba.jit
def indictors_style_all_market(coin_id, freq_length, cr):
    """
    全市场风格指标计算
    :param coin_id:
    :param freq_length:
    :param cr:
    :return indicators:
    """
    indicators = []

    date_range, interval = date_range_convert(None, freq_length)
    if date_range.get('min') is None or date_range.get('max') is None:
        return indicators

    engine = engine_currency()
    sql = "select coin_id, statistic_date, exchange_name,turnover_usd from coin_market where statistic_date BETWEEN '{}' and '{}' order by coin_id, statistic_date" \
        .format(date_range['min'], date_range['max'])
    df = pd.read_sql(sql, engine)
    df.fillna(0, inplace=True)
    if len(df) == 0:
        return indicators

    coin_ids = [coin_id] if coin_id else df.coin_id.drop_duplicates().tolist()

    for coin_id in coin_ids:
        df_coin = df[df.coin_id == coin_id]
        statistic_dates = np.array(df_coin.statistic_date.tolist())

        indicator = {
            'coin_id': coin_id,
            'concentrationratio': []
        }

        for statistic_date in statistic_dates:
            daily_concentrationratio = {}
            group = df_coin[df_coin.statistic_date == statistic_date].groupby('exchange_name').sum().sort_values(
                by=['turnover_usd'], ascending=False)

            for i in range(min(len(group),int(cr))):
                exchange = group.index[i].replace('.', '/')
                concentrationratio = group.turnover_usd[i] / group.turnover_usd.sum()
                if concentrationratio != 0:
                    daily_concentrationratio[exchange] = concentrationratio

            indicator['concentrationratio'].append(
                {"statistic_date": str(statistic_date), 'value': daily_concentrationratio})
        indicators.append(indicator)

    return indicators


def coin_component(statistic_date):
    """
    成分虚拟币
    :param statistic_date:
    :return component_coin:
    """
    statistic_date = dt.datetime.strptime(statistic_date, '%Y-%m-%d')
    date_ico = str(statistic_date - relativedelta(months=9))
    data_stable = str(statistic_date - relativedelta(months=6))
    engine = engine_currency()

    # 剔除上市时间不足9个月的虚拟币，未在交易所稳定交易6个月以上，小于等于5个交易所上线了该代币的情况
    component_stable = []
    sql = "select coin_id,ico_date from coin_info"
    df = pd.read_sql(sql, engine)
    coin_ids = df[df.ico_date < date_ico].coin_id.drop_duplicates().tolist()
    sql = "select coin_id,exchange_name,update_time_page from coin_market"
    df = pd.read_sql(sql, engine)
    df = df[df.update_time_page > data_stable]
    for coin_id in coin_ids:
        exchange_number = len(df[df.coin_id == coin_id].exchange_name.drop_duplicates().tolist())
        if exchange_number > 5:
            component_stable.append(coin_id)

    # 过去6个月市值连续超过3个月排名在100名以内
    component_rank = []
    sql = "select coin_id, statistic_date, market_value_usd from coin_trading where statistic_date BETWEEN '{}' and '{}' order by statistic_date" \
        .format(data_stable, statistic_date)
    df = pd.read_sql(sql, engine)
    statistic_dates = df.statistic_date.drop_duplicates().tolist()
    coin_ranks = []
    for statistic_date in statistic_dates:
        coin_rank = \
            df[df.statistic_date == statistic_date].sort_values(by="market_value_usd", ascending=False).head(100)[
                'coin_id'].tolist()
        coin_ranks.append(coin_rank)
    for coin_id in component_stable:
        n = 0
        flag = 1
        while n < (len(coin_ranks) - 90):
            for coin_rank in coin_ranks[n:n + 90]:
                if coin_id not in coin_rank:
                    flag = 0
            n += 1
        if flag:
            component_rank.append(coin_id)

    # 计算满足条件的虚拟币各交易所之间成交额的标准差，并取最小的30只虚拟币
    component_std = []
    sql = "select coin_id,statistic_date,exchange_name,turnover_usd from coin_market where statistic_date BETWEEN '{}' and '{}' order by statistic_date" \
        .format(data_stable, statistic_date)
    df = pd.read_sql(sql, engine)

    turnover_usd_std = []
    for coin_id in component_rank:
        turnover_usd_std.append(
            (coin_id, df[df.coin_id == coin_id].groupby("exchange_name").sum().turnover_usd.std()))
    for std in sorted(turnover_usd_std)[:30]:
        component_std.append(std[0])

    # 总市值和成交金额排序
    component_coin = component_std
    sql = "select coin_id, statistic_date, market_value_usd,turnover_usd from coin_trading where statistic_date BETWEEN '{}' and '{}' order by statistic_date" \
        .format(data_stable, statistic_date)
    df = pd.read_sql(sql, engine)
    coin_ids = df.coin_id.drop_duplicates().tolist()
    list_market_value = [df[df.coin_id == coin_id].market_value_usd.sum() for coin_id in coin_ids]
    list_turnover = [df[df.coin_id == coin_id].turnover_usd.sum() for coin_id in coin_ids]
    list_market_value.sort(reverse=True)
    list_turnover.sort(reverse=True)
    coin_rank = {}
    for coin_id in coin_ids:
        market_value_rank = list_market_value.index(df[df.coin_id == coin_id].market_value_usd.sum())
        turnover_rank = list_turnover.index(df[df.coin_id == coin_id].turnover_usd.sum())
        coin_rank[coin_id] = market_value_rank + turnover_rank

    for item in sorted(coin_rank.items(), key=lambda x: x[1]):
        if len(component_coin) >= 30:
            break
        else:
            if item[0] not in component_std:
                component_coin.append(item[0])

    return component_coin


# 每季度成分虚拟币
component_dict = {
    '2017-01-01': ['JRB100001', 'JRB100002', 'JRB100006', 'JRB100008', 'JRB100012', 'JRB100013', 'JRB100014',
                   'JRB100015', 'JRB100023', 'JRB100032', 'JRB100033', 'JRB100034', 'JRB100052', 'JRB100060',
                   'JRB100085', 'JRB100090', 'JRB100113', 'JRB100182', 'JRB100238', 'JRB100544', 'JRB100705',
                   'JRB100005', 'JRB100017', 'JRB100018', 'JRB100003', 'JRB100010', 'JRB100011', 'JRB100062',
                   'JRB100027', 'JRB100031'],
    '2017-04-01': ['JRB100001', 'JRB100002', 'JRB100006', 'JRB100008', 'JRB100012', 'JRB100013', 'JRB100014',
                   'JRB100015', 'JRB100023', 'JRB100026', 'JRB100032', 'JRB100033', 'JRB100034', 'JRB100041',
                   'JRB100043', 'JRB100052', 'JRB100060', 'JRB100085', 'JRB100090', 'JRB100113', 'JRB100138',
                   'JRB100182', 'JRB100238', 'JRB100005', 'JRB100003', 'JRB100017', 'JRB100018', 'JRB100042',
                   'JRB100048', 'JRB100062'],
    '2017-07-01': ['JRB100012', 'JRB100013', 'JRB100018', 'JRB100026', 'JRB100032', 'JRB100033', 'JRB100043',
                   'JRB100052', 'JRB100060', 'JRB100090', 'JRB100138', 'JRB100001', 'JRB100002', 'JRB100005',
                   'JRB100006', 'JRB100003', 'JRB100017', 'JRB100015', 'JRB100024', 'JRB100025', 'JRB100008',
                   'JRB100040', 'JRB100038', 'JRB100045', 'JRB100034', 'JRB100063', 'JRB100042', 'JRB100023',
                   'JRB100030', 'JRB100009'],
    '2017-10-01': ['JRB100013', 'JRB100025', 'JRB100026', 'JRB100033', 'JRB100038', 'JRB100042', 'JRB100043',
                   'JRB100052', 'JRB100120', 'JRB100001', 'JRB100002', 'JRB100003', 'JRB100006', 'JRB100004',
                   'JRB100018', 'JRB100005', 'JRB100017', 'JRB100010', 'JRB100011', 'JRB100012', 'JRB100009',
                   'JRB100015', 'JRB100022', 'JRB100024', 'JRB100021', 'JRB100019', 'JRB100014', 'JRB100040',
                   'JRB100008', 'JRB100034'],
    '2018-01-01': ['JRB100006', 'JRB100013', 'JRB100014', 'JRB100015', 'JRB100025', 'JRB100026', 'JRB100033',
                   'JRB100034', 'JRB100038', 'JRB100042', 'JRB100043', 'JRB100052', 'JRB100001', 'JRB100002',
                   'JRB100004', 'JRB100003', 'JRB100009', 'JRB100018', 'JRB100012', 'JRB100011', 'JRB100021',
                   'JRB100005', 'JRB100010', 'JRB100019', 'JRB100022', 'JRB100028', 'JRB100007', 'JRB100008',
                   'JRB100050', 'JRB100017'],
    '2018-04-01': ['JRB100001', 'JRB100002', 'JRB100003', 'JRB100006', 'JRB100008', 'JRB100011', 'JRB100012',
                   'JRB100013', 'JRB100014', 'JRB100015', 'JRB100018', 'JRB100021', 'JRB100022', 'JRB100023',
                   'JRB100025', 'JRB100026', 'JRB100032', 'JRB100033', 'JRB100034', 'JRB100038', 'JRB100040',
                   'JRB100041', 'JRB100042', 'JRB100043', 'JRB100048', 'JRB100054', 'JRB100057', 'JRB100066',
                   'JRB100085', 'JRB100117']
}


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

    while max(df_statistics.adjust_market_value) / sum(df_statistics.adjust_market_value) > 0.25:
        df_statistics.sort_values(by='adjust_market_value', ascending=False, inplace=True)
        for coin_id in df_statistics.index.tolist():
            weight = (df_statistics.loc[coin_id, 'adjust_market_value'] / sum(df_statistics.adjust_market_value))
            if weight > 0.25:
                df_statistics.loc[coin_id, 'ratio'] = df_statistics.loc[coin_id, 'ratio'] - 0.001
        df_statistics.adjust_market_value = df_statistics.market_value * df_statistics.ratio

    df_statistics['weight'] = df_statistics.adjust_market_value / sum(df_statistics.adjust_market_value)

    return df_statistics


def all_coin_index(update_date):
    """
    虚拟币指数(全)
    :param:
    :return indicators:
    """
    indicators = []
    engine = engine_currency()
    sql = "select coin_id, statistic_date, market_value_usd from coin_trading where statistic_date BETWEEN '{}' and '{}' order by statistic_date" \
        .format(dt.datetime.strptime('2017-1-1', '%Y-%m-%d'), dt.datetime.today())
    df = pd.read_sql(sql, engine)
    df.fillna('ffill', inplace=True)

    # 基期市值
    list_base = []
    component = component_dict['2017-01-01']
    for coin_id in component:
        market_value = df[df.coin_id == coin_id].iloc[0].market_value_usd
        list_base.append([coin_id, market_value, 1.])

    market_value_base = sum(index_statistics(list_base).adjust_market_value)

    # 成分比调整日期列表
    component_date = dt.datetime.strptime('2017-1-1', "%Y-%m-%d")
    component_date_list = []
    while (component_date - dt.datetime.today()).days < 0:
        component_date = component_date + relativedelta(months=3)
        component_date_list.append(str(component_date.date()))

    # 每日市值
    market_date_list = list(pd.date_range(start=update_date, end=dt.datetime.today()))
    for i in range(len(market_date_list)):
        statistic_date = market_date_list[i].strftime("%Y-%m-%d")

        if statistic_date in component_date_list:
            component = component_dict[statistic_date]

        list_daily = []
        for coin_id in component:
            df_coin = df[df.coin_id == coin_id]
            df_coin.index = pd.to_datetime(df_coin.statistic_date)
            del df_coin['statistic_date']
            df_coin = df_coin.reindex(index = market_date_list,method='ffill')
            market_value = df_coin.loc[statistic_date,'market_value_usd']
            list_daily.append([coin_id, market_value, 1.])

        market_value_daily = sum(index_statistics(list_daily).adjust_market_value)

        # 虚拟币指数
        coin_index_value = round(market_value_daily * 1000 / market_value_base, 4)
        indicators.append({"statistic_date": statistic_date, 'value': coin_index_value})
        pprint(indicators[-1])


    # df_coin_index_update = pd.DataFrame(indicators)
    #
    # all_coin_index = coin_index_mongo()['coin_index']
    # df_coin_index = pd.DataFrame(all_coin_index)
    # df_coin_index.index = pd.to_datetime(df_coin_index.statistic_date)


    return indicators


def interval_coin_index(begin, end):
    """
    指数和累计收益
    :param: statistic_date_begin, statistic_date_end:
    :return coin_index,coin_index_return:
    """
    all_coin_index = coin_index_mongo()['coin_index']
    df_coin_index = pd.DataFrame(all_coin_index)
    df_coin_index.index = pd.to_datetime(df_coin_index.statistic_date)
    df_coin_index = df_coin_index.loc[begin:end]

    coin_index = []
    coin_index_return = []

    for i in range(len(df_coin_index)):
        coin_index.append({
            "statistic_date": df_coin_index.iloc[i].statistic_date,
            'value': df_coin_index.iloc[i].value
        })
        coin_index_return.append({
            "statistic_date": df_coin_index.iloc[i].statistic_date,
            'value': df_coin_index.iloc[i].value / df_coin_index.iloc[0].value - 1
        })

    return coin_index, coin_index_return


if __name__ == "__main__":
    # pprint(coin_info(None))
    # pprint(indictors_return(None, 'y1'))
    # pprint(indictors_risk("JRB100001", 'y3'))
    # pprint(indictors_risk_adjust("JRB100001", 'y3'))
    #  pprint(indictors_style("JRB100017", 'm3'))
    # pprint(indictors_style_all_market("JRB100001", 'm3', 4))
    # pprint(coin_component('2018-1-1'))
    pprint(coin_concept('2018-4-28'))
    # pprint(all_coin_index('2018-5-1'))
    # pprint(interval_coin_index('2018-1-1', '2018-5-1'))

