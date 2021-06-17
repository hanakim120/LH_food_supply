import numpy as np
import pandas as pd

from konlpy.tag import Mecab

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.embedding import embedding, AutoEncoder, subword


def process(config):
    train_df = pd.read_csv('./data/train.csv', encoding='utf-8')
    test_df = pd.read_csv('./data/test.csv', encoding='utf-8')
    y = train_df[['중식계', '석식계']]
    sample = pd.read_csv('./data/sample_submission.csv', encoding='utf-8')

    # crawling
    temp = pd.read_csv('./data/temp.csv', encoding='utf-8')

    TRAIN_LENGTH = 1205

    train_df.drop(columns=['중식계', '석식계'], inplace=True)

    df = pd.concat([train_df, test_df], axis=0)
    df = pd.merge(df, temp, on='일자')

    if config.sep_date:
        df['month'] = df.일자.apply(lambda x : int(x[-5 :-3]))
        df['day'] = df.일자.apply(lambda x : int(x[-2 :]))
        df['week'] = df.day.apply(lambda x : x // 7)

        df.drop(columns=['일자', 'day'], inplace=True)
    else :
        df.일자 = pd.to_datetime(df.일자)
        df.rename(columns={'일자':'ds'}, inplace=True)

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

    if config.text == 'embedding' :
        breakfast = df.조식메뉴
        launch = df.중식메뉴
        dinner = df.석식메뉴

        tmp = embedding(config, breakfast, launch, dinner)

    if config.text == 'subword':
        tmp = subword(config)


    if config.text == 'tokenize' :
        menus = ['조식메뉴', '중식메뉴', '석식메뉴']
        for col in menus :
            tokenizer = Tokenizer(oov_token='<OOV>')
            tokenizer.fit_on_texts(df[col][:TRAIN_LENGTH])
            seq = tokenizer.texts_to_sequences(df[col])
            pad = pad_sequences(seq)

            if config.use_pca > 0 :
                pca = PCA(n_components=config.use_pca)
                pca.fit(pad[:TRAIN_LENGTH])
                pad = pca.transform(pad)

            length = len(pad[0])
            pad = pd.DataFrame(pad, columns=['{}_{}'.format(col, i) for i in range(length)])
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(pad)], axis=1)
        df.drop(columns=menus, inplace=True)

    # Normalize
    scaling_cols = ['본사정원수', '본사휴가자수', '본사출장자수',
                    '본사시간외근무명령서승인건수', '현본사소속재택근무자수']
    for col in scaling_cols :
        ms = MinMaxScaler()
        ms.fit(df[col][:TRAIN_LENGTH].values.reshape(-1, 1))
        df[col] = ms.transform(df[col].values.reshape(-1, 1))

    if config.text == 'embedding' or config.text == 'subword':
        new_df = pd.concat([df.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
        new_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴'], inplace=True)
    else :
        new_df = df.reset_index(drop=True)
    le = LabelEncoder()
    le.fit(new_df.요일.values[:TRAIN_LENGTH])
    new_df.요일 = le.transform(new_df.요일.values)

    train_df = new_df[:1000]
    valid_df = new_df[1000:1205]
    # train_df, valid_df, train_y, valid_y = train_test_split(df, train_y, test_size=0.2, random_state=0)
    test_df = new_df[TRAIN_LENGTH :]
    train_y = y[:1000]
    valid_y = y[1000:]

    return train_df, valid_df, test_df, train_y, valid_y, sample


def autoencoding(train_df, test_df, TARGET=1, verbose=2) :
    TRAIN_LENGTH = 1205

    df = pd.concat([train_df, test_df], axis=0)
    text_df = df[['조식메뉴', '중식메뉴', '석식메뉴']]

    mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

    for i in range(len(text_df.조식메뉴)) :
        text_df.조식메뉴[i] = ','.join(text_df.조식메뉴[i].split())
        text_df.조식메뉴[i] = ' '.join(mecab.morphs(text_df.조식메뉴[i]))
    for i in range(len(text_df.중식메뉴)) :
        text_df.중식메뉴[i] = ','.join(text_df.중식메뉴[i].split())
        text_df.중식메뉴[i] = ' '.join(mecab.morphs(text_df.중식메뉴[i]))
    for i in range(len(text_df.석식메뉴)) :
        text_df.석식메뉴[i] = ','.join(text_df.석식메뉴[i].split())
        text_df.석식메뉴[i] = ' '.join(mecab.morphs(text_df.석식메뉴[i]))

    vect = CountVectorizer()
    vect.fit(text_df['조식메뉴'].values[:TRAIN_LENGTH])
    breakfast = vect.transform(text_df['조식메뉴'].values).toarray()
    vect.fit(text_df['중식메뉴'].values[:TRAIN_LENGTH])
    launch = vect.transform(text_df['중식메뉴'].values).toarray()
    vect.fit(text_df['석식메뉴'].values[:TRAIN_LENGTH])
    dinner = vect.transform(text_df['석식메뉴'].values).toarray()

    enc_df = pd.DataFrame()

    i = 1
    for menu in [breakfast, launch, dinner] :
        print('+' * 10, 'Train {}'.format(i), '+' * 10)
        train_X, valid_X = train_test_split(menu[:TRAIN_LENGTH],
                                            test_size=0.1,
                                            shuffle=True,
                                            random_state=0)
        STEP = (menu.shape[1] - TARGET) // 5
        INPUT = menu.shape[1]

        model = AutoEncoder(INPUT, STEP)
        model.compile(loss='mse', optimizer='adam', metrics='mse')
        model.fit(train_X, train_X,
                  validation_data=(valid_X, valid_X),
                  epochs=1000,
                  callbacks=tf.keras.callbacks.EarlyStopping(patience=10),
                  verbose=verbose)
        result = model.encoder(menu)
        result = np.array(result)

        enc_df = pd.concat([enc_df, pd.DataFrame(result)], axis=1)
        i += 1
        print()

    train_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴'], inplace=True)
    test_df.drop(columns=['조식메뉴', '중식메뉴', '석식메뉴'], inplace=True)

    train_df_enc = pd.concat([train_df, enc_df[:TRAIN_LENGTH]], axis=1)
    test_df_enc = pd.concat([test_df, enc_df[TRAIN_LENGTH :]], axis=1)

    column_names = list(train_df_enc.columns[:8]) + [i for i in range(enc_df.shape[1])]
    train_df_enc.columns = column_names
    test_df_enc.columns = column_names

    return train_df_enc, test_df_enc


def get_data(config) :
    train_df, valid_df, test_df, train_y, valid_y, sample = process(config)
    if config.text == 'autoencode' :
        train_df, test_df = autoencoding(train_df, test_df, TARGET=config.dim)

    return train_df, valid_df, test_df, train_y, valid_y, sample