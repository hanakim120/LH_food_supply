import re

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup


def get_weather() :
    print('crawling weather data, please wait...')
    df = pd.DataFrame(np.zeros((1, 11)))

    for year in range(2016, 2022) :
        for month in range(1, 13) :
            response = requests.get(
                'https://www.weatheri.co.kr/bygone/pastDB_tmp.php?jijum_id=192&start={}-{}-01&end=2021-4-31'.format(
                    year, month))
            dom = BeautifulSoup(response.content, 'html.parser')
            elements = dom.select('td tr > td td')

            row = []
            cnt = 0

            for i in range(len(elements)) :
                text = elements[i].getText()
                text = re.sub('\n', '', text)
                text = re.sub('\t', '', text)
                row.append(text)
                cnt += 1
                if cnt == 11 :
                    row = pd.DataFrame(np.array(row).reshape(1, -1))
                    df = pd.concat([df, row], axis=0)
                    row = []
                    cnt = 0

    df = df.reset_index(drop=True)
    return df

def main():
    df = get_weather()

    print('processing data..')
    df[0] = df[0].str.replace('년 ', '-')
    df[0] = df[0].str.replace('월', '-')
    df[0] = df[0].str.replace('일 ', '')
    df[0] = df[0].str.replace(r'\([ㄱ-힣\-]\)', '')

    df.columns = ['일자', '평균기온', '최고기온', '최저기온', '강수량', '적설량', '풍속', '습도', '운량', '일조시간', '날씨']
    df.reset_index(drop=True, inplace=True)

    for i in range(df.shape[0]) :
        if df.loc[i, '일자'] == '날짜' :
            df.drop(index=[i], inplace=True)

    df.reset_index(drop=True, inplace=True)
    for i in range(df.shape[0]) :
        if df.loc[i, '강수량'] == '-' :
            df.loc[i, '강수량'] = 0.
        if df.loc[i, '적설량'] == '-' :
            df.loc[i, '적설량'] = 0.
    df.일자 = pd.to_datetime(df.일자)

    df.to_csv('./data/temp.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()