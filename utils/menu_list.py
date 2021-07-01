import numpy as np
import pandas as pd

from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer

def make_menulist():
    df = pd.read_csv('./data/train.csv', encoding='utf-8')

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

    corpus = pd.concat([df.조식메뉴, df.중식메뉴, df.석식메뉴], axis=0).reset_index(drop=True)
    #
    mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')
    for i in range(corpus.shape[0]):
        corpus.iloc[i] = ' '.join(mecab.morphs(corpus.iloc[i]))
    print(corpus.tail())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    print(len(tokenizer.word_docs))
    tmp = pd.DataFrame(np.zeros((2960, 1)))

    for i, v in enumerate(tokenizer.word_docs.keys()):
        tmp.loc[i] = v

    tmp.to_csv('menu.csv', index=True, encoding='utf-8-sig')


if __name__ == '__main__':
    make_menulist()
