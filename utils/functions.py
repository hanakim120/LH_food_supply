import numpy as np
import pandas as pd
import statsmodels.api as sm

def process_holiday(train_df, test_df, config):
    # 공휴일 전후
    train_df['공휴일전후'] = 0
    test_df['공휴일전후'] = 0
    train_df['공휴일전후'][17] = 1
    train_df['공휴일전후'][3] = 1
    train_df['공휴일전후'][62] = 1
    train_df['공휴일전후'][131] = 1
    train_df['공휴일전후'][152] = 1
    train_df['공휴일전후'][226] = 1
    train_df['공휴일전후'][221] = 1
    train_df['공휴일전후'][224] = 1
    train_df['공휴일전후'][245] = 1
    train_df['공휴일전후'][310] = 1  # 2
    train_df['공휴일전후'][311] = 1
    train_df['공휴일전후'][309] = 1
    train_df['공휴일전후'][330] = 1
    train_df['공휴일전후'][379] = 1
    train_df['공휴일전후'][467] = 1
    train_df['공휴일전후'][470] = 1
    train_df['공휴일전후'][502] = 1
    train_df['공휴일전후'][565] = 1
    train_df['공휴일전후'][623] = 1
    train_df['공휴일전후'][651] = 1
    train_df['공휴일전후'][705] = 1
    train_df['공휴일전후'][709] = 1
    train_df['공휴일전후'][815] = 1
    train_df['공휴일전후'][864] = 1
    train_df['공휴일전후'][950] = 1
    train_df['공휴일전후'][951] = 1
    train_df['공휴일전후'][953] = 1
    train_df['공휴일전후'][954] = 1
    train_df['공휴일전후'][955] = 1
    train_df['공휴일전후'][971] = 1  # 2
    train_df['공휴일전후'][1038] = 1
    train_df['공휴일전후'][1099] = 1
    train_df['공휴일전후'][1129] = 1
    train_df['공휴일전후'][1187] = 1
    test_df['공휴일전후'][10] = 1
    test_df['공휴일전후'][20] = 1

    if config.holiday_length:
        train_df['공휴일길이'] = 0
        test_df['공휴일길이'] = 0
        train_df['공휴일길이'][17] = 1  # 삼일절 전
        train_df['공휴일길이'][6] = 2  # 설 연휴 전
        train_df['공휴일길이'][47] = 1  # 총선 전
        train_df['공휴일길이'][62] = 4  # 어린이 날 전
        train_df['공휴일길이'][82] = 4  # 현충일 연휴 전(금요일)
        train_df['공휴일길이'][131] = 3  # 광복절 연휴 전(금요일)
        train_df['공휴일길이'][152] = 5  # 한가위 연휴 전
        train_df['공휴일길이'][162] = 3  # 개천절 연휴 전(금요일)
        train_df['공휴일길이'][226] = 2  # 연말
        train_df['공휴일길이'][221] = 2  # 크리스마스 전(금요일)
        train_df['공휴일길이'][245] = 4  # 설 연휴 전
        train_df['공휴일길이'][310] = 3  # 어린이 날 연휴 전(금요일)
        train_df['공휴일길이'][309] = 1  # 석가탄신일 전
        train_df['공휴일길이'][330] = 1  # 현충일 전
        train_df['공휴일길이'][379] = 1  # 광복절 전
        train_df['공휴일길이'][412] = 11  # 추석연휴 전
        train_df['공휴일길이'][466] = 3  # 성탄절 연휴 전(금요일)
        train_df['공휴일길이'][470] = 3  # 연말
        train_df['공휴일길이'][502] = 4  # 설 연휴 전
        train_df['공휴일길이'][510] = 3  # 삼일절 전
        train_df['공휴일길이'][565] = 1  # 석가탄신일 전
        train_df['공휴일길이'][575] = 1  # 현충일 전
        train_df['공휴일길이'][623] = 1  # 광복절 전
        train_df['공휴일길이'][650] = 2  # 추석 연휴 전
        train_df['공휴일길이'][651] = 16  # 한글날 전
        train_df['공휴일길이'][705] = 1  # 크리스마스 이브
        train_df['공휴일길이'][709] = 1  # 연말
        train_df['공휴일길이'][732] = 5  # 설연휴 전
        train_df['공휴일길이'][748] = 3  # 삼일절 연휴 전
        train_df['공휴일길이'][792] = 3  # 어린이날 연휴 전
        train_df['공휴일길이'][814] = 1  # 현충일 전
        train_df['공휴일길이'][863] = 1  # 광복절 전
        train_df['공휴일길이'][882] = 4  # 추석 연휴 전
        train_df['공휴일길이'][894] = 1  # 개천절 전
        train_df['공휴일길이'][897] = 1  # 한글날 전
        train_df['공휴일길이'][951] = 1  # 크리스마스 전
        train_df['공휴일길이'][955] = 1  # 연말
        train_df['공휴일길이'][971] = 4  # 설 연휴
        train_df['공휴일길이'][1027] = 1  # 국회의원선거 전
        train_df['공휴일길이'][1037] = 4  # 석가탄신일 연휴 전
        train_df['공휴일길이'][1038] = 1  # 어린이날 전
        train_df['공휴일길이'][1129] = 7  # 추석연휴 전
        train_df['공휴일길이'][1133] = 3  # 한글날 연휴 전
        train_df['공휴일길이'][1187] = 10  # 성탄절 연휴 전

        test_df['공휴일길이'][10] = 4  # 설연휴전
        test_df['공휴일길이'][20] = 3  # 삼일절 연휴 전

    return train_df, test_df


def get_month_average(train):
    월1 = 0
    월2 = 0
    월3 = 0
    월4 = 0
    월5 = 0
    월6 = 0
    월7 = 0
    월8 = 0
    월9 = 0
    월10 = 0
    월11 = 0
    월12 = 0

    월1cnt = 0
    월2cnt = 0
    월3cnt = 0
    월4cnt = 0
    월5cnt = 0
    월6cnt = 0
    월7cnt = 0
    월8cnt = 0
    월9cnt = 0
    월10cnt = 0
    월11cnt = 0
    월12cnt = 0
    arrays = []
    for col in ['중식계', '석식계'] :
        for i in range(train.shape[0]) :
            if train.loc[i, 'month'] == 1 :
                월1 += train.loc[i, col]
                월1cnt += 1
            elif train.loc[i, 'month'] == 2 :
                월2 += train.loc[i, col]
                월2cnt += 1
            elif train.loc[i, 'month'] == 3 :
                월3 += train.loc[i, col]
                월3cnt += 1
            elif train.loc[i, 'month'] == 4 :
                월4 += train.loc[i, col]
                월4cnt += 1
            elif train.loc[i, 'month'] == 5 :
                월5 += train.loc[i, col]
                월5cnt += 1
            elif train.loc[i, 'month'] == 6 :
                월6 += train.loc[i, col]
                월6cnt += 1
            elif train.loc[i, 'month'] == 7 :
                월7 += train.loc[i, col]
                월7cnt += 1
            elif train.loc[i, 'month'] == 8 :
                월8 += train.loc[i, col]
                월8cnt += 1
            elif train.loc[i, 'month'] == 9 :
                월9 += train.loc[i, col]
                월9cnt += 1
            elif train.loc[i, 'month'] == 10 :
                월10 += train.loc[i, col]
                월10cnt += 1
            elif train.loc[i, 'month'] == 11 :
                월11 += train.loc[i, col]
                월11cnt += 1
            elif train.loc[i, 'month'] == 12 :
                월12 += train.loc[i, col]
                월12cnt += 1

        월1 /= 월1cnt
        월2 /= 월2cnt
        월3 /= 월3cnt
        월4 /= 월4cnt
        월5 /= 월5cnt
        월6 /= 월6cnt
        월7 /= 월7cnt
        월8 /= 월8cnt
        월9 /= 월9cnt
        월10 /= 월10cnt
        월11 /= 월11cnt
        월12 /= 월12cnt

        array = [월1, 월2, 월3, 월4, 월5, 월6, 월7, 월8, 월9, 월10, 월11, 월12]
        arrays.append(array)

    return arrays

def get_week_average(train):
    주1 = 0
    주2 = 0
    주3 = 0
    주4 = 0
    주5 = 0

    주1cnt = 0
    주2cnt = 0
    주3cnt = 0
    주4cnt = 0
    주5cnt = 0

    arrays = []
    for col in ['중식계', '석식계'] :
        for i in range(train.shape[0]) :
            if train.loc[i, 'week'] == 0 :
                주1 += train.loc[i, col]
                주1cnt += 1
            elif train.loc[i, 'week'] == 1 :
                주2 += train.loc[i, col]
                주2cnt += 1
            elif train.loc[i, 'week'] == 2 :
                주3 += train.loc[i, col]
                주3cnt += 1
            elif train.loc[i, 'week'] == 3 :
                주4 += train.loc[i, col]
                주4cnt += 1
            elif train.loc[i, 'week'] == 4 :
                주5 += train.loc[i, col]
                주5cnt += 1

        주1 /= 주1cnt
        주2 /= 주2cnt
        주3 /= 주3cnt
        주4 /= 주4cnt
        주5 /= 주5cnt

        array = [주1, 주2, 주3, 주4, 주5]
        arrays.append(array)

    return arrays

def drop_outlier(train_df, valid_df, train_y, valid_y, config):
    # _train_df = pd.concat([train_df, valid_df], axis=0)
    # _train_y = pd.concat([train_y, valid_y], axis=0)
    # if config.outlier == 'fox' :
    #     idx1 = [10, 35, 44, 107, 109, 127, 152, 196, 207, 225, 238, 267, 285, 304,
    #             311, 341, 344, 351, 353, 382, 384, 401, 403, 426, 434, 466, 492, 530,
    #             537, 553, 642, 665, 692, 704, 716, 724, 742, 743, 762, 779, 787, 796,
    #             799, 805, 807, 819, 825, 852, 881, 882, 901, 903, 920, 954, 990, 998,
    #             1002, 1004, 1017, 1031, 1041, 1056, 1058, 1097, 1106, 1119, 1133, 1153, 1167, 1168,
    #             1176, 1179, 1194, 1197]
    #     idx2 = [9, 16, 25, 41, 72, 78, 84, 152, 169, 201, 221, 259, 260, 264,
    #             274, 311, 320, 351, 353, 365, 382, 400, 403, 404, 448, 455, 464, 480,
    #             484, 492, 497, 507, 535, 552, 594, 614, 644, 659, 673, 675, 681, 702,
    #             704, 713, 716, 724, 727, 737, 739, 745, 746, 751, 769, 787, 801, 825,
    #             835, 852, 854, 864, 870, 874, 881, 887, 894, 895, 920, 936, 948, 990,
    #             1007, 1017, 1058, 1092, 1097, 1098, 1119, 1132, 1140, 1143, 1145, 1153, 1164, 1165,
    #             1168, 1176, 1179]
    # elif config.outlier == 'simple' :
    #     idx1 = [492]
    #     idx2 = [492]
    # else :
    #     raise NotImplementedError
    model = sm.OLS(train_y.중식계.reset_index(drop=True), sm.add_constant(train_df).reset_index(drop=True))
    result = model.fit()
    influence = result.get_influence()
    cooks_d2, pvals = influence.cooks_distance
    k = influence.k_vars
    fox_cr = 4 / (len(train_y) - k - 1)

    idx = np.where(cooks_d2 > fox_cr)[0]


    lunch_idx = list(set(range(len(train_df))).difference(idx))
    lunch_df = train_df.iloc[lunch_idx, :].reset_index(drop=True)
    lunch_y = train_y.iloc[lunch_idx, :].reset_index(drop=True)

    model = sm.OLS(train_y.석식계.reset_index(drop=True), sm.add_constant(train_df).reset_index(drop=True))
    result = model.fit()
    influence = result.get_influence()
    cooks_d2, pvals = influence.cooks_distance
    k = influence.k_vars
    fox_cr = 4 / (len(train_y) - k - 1)

    idx = np.where(cooks_d2 > fox_cr)[0]

    dinner_idx = list(set(range(len(train_df))).difference(idx))
    dinner_df = train_df.iloc[dinner_idx, :].reset_index(drop=True)
    dinner_y = train_y.iloc[dinner_idx, :].reset_index(drop=True)

    return lunch_df, lunch_y, dinner_df, dinner_y, valid_df, valid_y