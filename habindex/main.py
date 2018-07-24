import pandas as pd
from func import all_coin_index,back_test


if __name__ == '__main__':
    # 计算HAB30历史数据

    index_data = all_coin_index()
    index_data = pd.DataFrame(index_data)
    index_data.to_csv('d:/habo/data/HAB30/HAB_index_a0.25.csv',index = False)

