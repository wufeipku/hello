import pandas as pd
from func import back_test


if __name__ == '__main__':
#回测
    data = pd.read_csv('d:/habo/data/HAB30/HAB_index_a0.25.csv')
    back_test(data)