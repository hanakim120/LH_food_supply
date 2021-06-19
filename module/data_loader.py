import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.embedding import embedding, autoencoding

def process(config):
    train_df = pd.read_csv('./data/train.csv', encoding='utf-8')
    test_df = pd.read_csv('./data/test.csv', encoding='utf-8')
    sample = pd.read_csv('./data/sample_submission.csv', encoding='utf-8')

    TRAIN_LENGTH = 1205

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
    train_df['공휴일전후'][310] = 1 # 2
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
    train_df['공휴일전후'][971] = 1 #2
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

    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    df = pd.get_dummies(df, columns=['공휴일전후'])

    y = train_df[['중식계', '석식계']]

    df['출근'] = df['본사정원수'] - (df['본사휴가자수'] + df['본사출장자수'] + df['현본사소속재택근무자수'])
    df['휴가비율'] = df['본사휴가자수'] / df['본사정원수']
    df['출장비율'] = df['본사출장자수'] / df['본사정원수']
    df['야근비율'] = df['본사시간외근무명령서승인건수'] / df['출근']
    df['재택비율'] = df['현본사소속재택근무자수'] / df['본사정원수']

    df.drop(columns=['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수'], inplace=True)

    temp = pd.read_csv('./data/temp.csv', encoding='utf-8')
    df = pd.merge(df, temp, on='일자')

    df['year'] = df.일자.apply(lambda x : int(x[:4]))
    df['month'] = df.일자.apply(lambda x : int(x[-5 :-3]))
    df['day'] = df.일자.apply(lambda x : int(x[-2 :]))
    df['week'] = df.day.apply(lambda x : x // 7)

    df.drop(columns=['일자'], inplace=True)

    columns = ['조식메뉴', '중식메뉴', '석식메뉴']
    for col in columns :
        df[col] = df[col].str.replace('/', ' ')
        df[col] = df[col].str.replace(r'([<]{1}[ㄱ-힣\:\,\.\/\-A-Za-z 0-9]*[>]{1})', '')
        df[col] = df[col].str.replace(r'([＜]{1}[ㄱ-힣\:\,\.\/\-A-Za-z 0-9]*[＞]{1})', '')
        df[col] = df[col].str.replace(r'([(]{1}[ㄱ-힣\:\,\.\/\-A-Za-z 0-9]*[)]{1})', '')
        df[col] = df[col].str.replace(r'[ ]{2, }', ' ')
        df[col] = df[col].str.replace('\(New\)', '')
        df[col] = df[col].str.replace('\(NeW\)', '')
        df[col] = df[col].str.replace(r'[D]{1}', '')
        df[col] = df[col].str.replace(r'[S]{1}', '')
        df[col] = df[col].str.replace('\(쌀:국내산,돈육:국내', '')
        df[col] = df[col].str.replace('고추가루:중국산\)', '')
        df[col] = df[col].str.replace('*', ' ')
        df[col] = df[col].str.replace('[(]만두 고추 통계란[)]', '')
        df[col] = df[col].str.replace('[(]모둠튀김 양념장[)]', '')
        df[col] = df[col].apply(lambda x : x.strip())

    # Normalize
    scaling_cols = ['본사시간외근무명령서승인건수', '출근', '휴가비율', '야근비율', '재택비율', '출장비율', 'year', 'day']
    for col in scaling_cols :
        ms = MinMaxScaler()
        ms.fit(df[col][:TRAIN_LENGTH].values.reshape(-1, 1))
        df[col] = ms.transform(df[col].values.reshape(-1, 1))

    le = LabelEncoder()
    le.fit(df.요일.values[:TRAIN_LENGTH])
    df.요일 = le.transform(df.요일.values)

    np.random.seed(0)
    idx = np.random.permutation(TRAIN_LENGTH)
    train_idx = idx[:1000]
    valid_idx = idx[1000:]

    train_df = df.iloc[train_idx, :]
    valid_df = df.iloc[valid_idx, :]
    test_df = df.iloc[TRAIN_LENGTH:, :]
    train_y = y.iloc[train_idx, :]
    valid_y = y.iloc[valid_idx, :]

    train_df.drop(columns=['중식계', '석식계'], inplace=True)
    valid_df.drop(columns=['중식계', '석식계'], inplace=True)
    test_df.drop(columns=['중식계', '석식계'], inplace=True)

    if config.text == 'embedding' :
        train_tmp, valid_tmp, test_tmp = embedding(config, train_df, valid_df, test_df)
    elif config.text == 'autoencoder' :
        train_tmp, valid_tmp, test_tmp = autoencoding(config, train_df, valid_df, test_df)
    else:
        print('ERROR: INVALID EMBEDDING METHOD, RUNTIME TERMINATED')
        quit()

    train_df = pd.concat([train_df.reset_index(drop=True), train_tmp.reset_index(drop=True)], axis=1)
    train_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴'], inplace=True)

    valid_df = pd.concat([valid_df.reset_index(drop=True), valid_tmp.reset_index(drop=True)], axis=1)
    valid_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴'], inplace=True)

    test_df = pd.concat([test_df.reset_index(drop=True), test_tmp.reset_index(drop=True)], axis=1)
    test_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴'], inplace=True)

    print('=' * 10, 'MESSAGE: DATA LOADED SUCCESSFULLY', '=' * 10)
    print('|TRAIN| : {} |VALID| : {} |TEST| : {}'.format(train_df.shape, valid_df.shape, test_df.shape))

    return train_df, valid_df, test_df, train_y, valid_y, sample


def get_data(config) :
    train_df, valid_df, test_df, train_y, valid_y, sample = process(config)

    return train_df, valid_df, test_df, train_y, valid_y, sample