import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.functions import get_month_average, process_holiday
from utils.embedding import embedding, autoencoding, menu_embedding


def get_ratio(df):
    df['출근'] = df['본사정원수'] - (df['본사휴가자수'] + df['본사출장자수'] + df['현본사소속재택근무자수'])
    df['휴가비율'] = df['본사휴가자수'] / df['본사정원수']
    df['출장비율'] = df['본사출장자수'] / df['본사정원수']
    df['야근비율'] = df['본사시간외근무명령서승인건수'] / df['출근']
    df['재택비율'] = df['현본사소속재택근무자수'] / df['본사정원수']

    df.drop(columns=['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수', '본사시간외근무명령서승인건수'], inplace=True)

    return df

def get_date(df):
    df['year'] = df.일자.apply(lambda x : int(x[:4]))
    df['month'] = df.일자.apply(lambda x : int(x[-5 :-3]))
    df['day'] = df.일자.apply(lambda x : int(x[-2 :]))
    df['week'] = df.day.apply(lambda x : x // 7)

    return df

def process_text(df):
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
        if col == '석식메뉴' :
            df[col] = df[col].str.replace('자기개발의날', '')
            df[col] = df[col].str.replace('자기계발의날', '')
            df[col] = df[col].str.replace('가정의날', '')

    return df

def process(config):
    train_df = pd.read_csv('./data/train.csv', encoding='utf-8')
    y = train_df[['중식계', '석식계']]
    test_df = pd.read_csv('./data/test.csv', encoding='utf-8')
    sample = pd.read_csv('./data/sample_submission.csv', encoding='utf-8')
    TRAIN_LENGTH = 1205

    train_df, test_df = process_holiday(train_df, test_df)
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    df = get_date(get_ratio(df))

    # weather =================================
    # SRC: https://www.weatheri.co.kr/index.php
    weather_columns = ['일자', '평균기온', '강수량', '습도', '풍속']
    weather = pd.read_csv('./data/temp.csv', encoding='utf-8')[weather_columns]
    df = pd.merge(df, weather, on='일자')

    df['불쾌지수'] = 0
    df['체감온도'] = 0

    df.reset_index(drop=True, inplace=True)
    for i in range(df.shape[0]):
        df.loc[i, '불쾌지수'] = (9. * df.loc[i, '평균기온'] / 5.) - (0.55 * (1 - df.loc[i, '습도']) * ((9. * df.loc[i, '평균기온'] / 5.) - 26)) + 32
        df.loc[i, '체감온도'] = 13.12 + (0.6215 * df.loc[i, '평균기온']) - (11.37 * (df.loc[i, '풍속'] ** 0.16)) + (0.3965 * (df.loc[i, '풍속'] ** 0.16) * df.loc[i, '평균기온'])

    df.drop(columns=['평균기온', '습도', '풍속'], inplace=True)
    # weather =================================

    df['요일점심평균'] = 0
    for i in range(df.shape[0]):
        if df.loc[i, '요일'] == '월' :
            df.loc[i, '요일점심평균'] = 1144.331950207469
        if df.loc[i, '요일'] == '화' :
            df.loc[i, '요일점심평균'] = 925.6208333333333
        if df.loc[i, '요일'] == '수' :
            df.loc[i, '요일점심평균'] = 905.2133891213389
        if df.loc[i, '요일'] == '목' :
            df.loc[i, '요일점심평균'] = 823.9918032786885
        if df.loc[i, '요일'] == '금' :
            df.loc[i, '요일점심평균'] = 653.6099585062241

    df['요일저녁평균'] = 0
    for i in range(df.shape[0]):
        if df.loc[i, '요일'] == '월' :
            df.loc[i, '요일저녁평균'] = 271.8409376560321
        if df.loc[i, '요일'] == '화' :
            df.loc[i, '요일저녁평균'] = 261.99296006944445
        if df.loc[i, '요일'] == '수' :
            df.loc[i, '요일저녁평균'] = 183.70128324083964
        if df.loc[i, '요일'] == '목' :
            df.loc[i, '요일저녁평균'] = 241.889327465735
        if df.loc[i, '요일'] == '금' :
            df.loc[i, '요일저녁평균'] = 203.84566381432825

    averages = get_month_average(df[:TRAIN_LENGTH])

    df['월점심평균'] = 0
    for i in range(df.shape[0]):
        df.loc[i, '월점심평균'] = averages[0][df.loc[i, 'month'] - 1]
    df['월저녁평균'] = 0
    for i in range(df.shape[0]) :
        df.loc[i, '월저녁평균'] = averages[1][df.loc[i, 'month'] - 1]

    # corona ===========================================
    # SRC: https://kdx.kr/data/view/25918
    corona = pd.read_csv('./data/corona/Covid19SidoInfState.csv')
    corona = corona.loc[corona.gubun == '경남']
    corona.stdDay = corona.stdDay.str.replace('년 ', '-')
    corona.stdDay = corona.stdDay.str.replace('월 ', '-')
    corona.stdDay = corona.stdDay.str.replace('일', '')
    corona.stdDay = corona.stdDay.str.replace('[ ][0-9]*시', '')
    df.일자 = pd.to_datetime(df.일자)
    corona.stdDay = pd.to_datetime(corona.stdDay)
    corona.rename(columns={'stdDay' : '일자',
                           'defCnt' : '확진자수',
                           'incDec' : '전일대비증감',
                           'deathCnt' : '사망자수'},
                  inplace=True)
    si = SimpleImputer(fill_value=0, strategy='constant')

    corona_columns = ['전일대비증감']
    for col in corona_columns:
        corona[col] = si.fit_transform(corona[col].values.reshape(-1, 1))
    df = pd.merge(df, corona[['일자'] + corona_columns], on='일자', how='left')
    df.drop_duplicates('일자', keep='first', inplace=True)
    for col in corona_columns:
        df[col] = si.fit_transform(df[col].values.reshape(-1, 1))
    # corona ===========================================

    df = process_text(df)

    # Normalize
    scaling_cols = ['공휴일길이',
                    '출근', '휴가비율', '야근비율', '재택비율', '출장비율',
                    'year',
                    '강수량', '불쾌지수', '체감온도',
                    '요일점심평균', '요일저녁평균', '월점심평균', '월저녁평균'] + corona_columns
    for col in scaling_cols :
        ms = MinMaxScaler()
        ms.fit(df[col][:TRAIN_LENGTH].values.reshape(-1, 1))
        df[col] = ms.transform(df[col].values.reshape(-1, 1))

    le = LabelEncoder()
    le.fit(df.요일.values[:TRAIN_LENGTH])
    df.요일 = le.transform(df.요일.values)

    if config.dummy_cat or config.model == 'rf' or config.model == 'randomforest' or config.model == 'reg':
        df = pd.get_dummies(df, columns=['요일', '공휴일전후', 'month', 'week'])

    np.random.seed(config.seed)
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

    if not config.drop_text:
        if config.text == 'embedding' :
            train_tmp, valid_tmp, test_tmp = embedding(config, train_df, valid_df, test_df)
        elif config.text == 'autoencoder' :
            train_tmp, valid_tmp, test_tmp = autoencoding(config, train_df, valid_df, test_df)
        elif config.text == 'menu':
            train_tmp, valid_tmp, test_tmp = menu_embedding(train_df, valid_df, test_df)
        else:
            print('ERROR: INVALID EMBEDDING METHOD, RUNTIME TERMINATED')
            quit()

        train_df = pd.concat([train_df.reset_index(drop=True), train_tmp.reset_index(drop=True)], axis=1)
        valid_df = pd.concat([valid_df.reset_index(drop=True), valid_tmp.reset_index(drop=True)], axis=1)
        test_df = pd.concat([test_df.reset_index(drop=True), test_tmp.reset_index(drop=True)], axis=1)

    train_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴', '일자'], inplace=True)
    valid_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴', '일자'], inplace=True)
    test_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴', '일자'], inplace=True)

    # train_df = train_df.reset_index(drop=True).drop(index=[870])
    # train_y = train_y.reset_index(drop=True).drop(index=[870])

    print('=' * 10, 'MESSAGE: DATA LOADED SUCCESSFULLY', '=' * 10)
    print('|TRAIN| : {} |VALID| : {} |TEST| : {}'.format(train_df.shape, valid_df.shape, test_df.shape))
    print('Missing values: {}'.format(np.isnan(train_df).sum().sum()))

    return train_df, valid_df, test_df, train_y, valid_y, sample


def get_data(config) :
    train_df, valid_df, test_df, train_y, valid_y, sample = process(config)

    return train_df, valid_df, test_df, train_y, valid_y, sample

if __name__ == '__main__':
    config = {'text' : 'embedding',
              'holiday_length' : True,
              'model' : 'reg',
              'seed' : 0,
              'drop_text' : False}
    train_df, valid_df, test_df, train_y, valid_y, sample = get_data(config)
    train_df.to_csv('train_df.csv', index=False, encoding='utf-8-sig')
    valid_df.to_csv('valid_df.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test_df.csv', index=False, encoding='utf-8-sig')
    train_y.to_csv('train_y.csv', index=False, encoding='utf-8-sig')
    valid_y.to_csv('valid_y.csv', index=False, encoding='utf-8-sig')