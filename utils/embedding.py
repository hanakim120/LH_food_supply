import re
from argparse import Namespace

import numpy as np
import pandas as pd

import fasttext
from konlpy.tag import Mecab

import tensorflow as tf
from tensorflow.keras.layers import *
from sklearn.feature_extraction.text import CountVectorizer

from utils.dummy import main


TRAIN_LENGTH = 1205


def pretext(text) :
    text = text.rstrip()
    text = re.sub('▁', '', text)
    return text

def get_fasttext_model(config, train_df, save=False):
    print('"{}" DOES NOT EXIST. START TRAINING MODEL'.format(config.fasttext_model_fn))
    corpus = np.concatenate([train_df.조식메뉴.values,
                             train_df.중식메뉴.values,
                             train_df.석식메뉴.values], axis=0)
    with open('./data/corpus.train.txt', 'w', -1, encoding='utf-8') as f :
        f.write('\n'.join(corpus))

    if config.dummy_corpus :
        args = {'load_fn' : './data/corpus.train.txt',
                'save_fn' : './data/corpus.train.dummy.txt',
                'iter' : 10,
                'verbose' : 300}
        args = Namespace(**args)
        main(args)

    model = fasttext.train_unsupervised(
        './data/corpus.train.dummy.txt' if config.dummy_corpus else './data/corpus.train.txt',
        dim=config.dim,
        ws=config.window_size,
        epoch=config.fasttext_epoch,
        min_count=config.min_count,
        minn=config.min_ngram,
        maxn=config.max_ngram,
        )
    if save:
        model.save_model('./data/{}'.format(config.fasttext_model_fn))
    return model

def embedding(config, train_df, valid_df, test_df):
    embedding_features = []

    if config.pretrained:
        print('USE PRETRAINED MODEL: cc.ko.300.bin')
        model = fasttext.load_model('cc.ko.300.bin')
        fasttext.util.reduce_model(model, config.dim)

    else:
        if config.fasttext_model_fn is not None:
            try:
                model = fasttext.load_model('./data/{}'.format(config.fasttext_model_fn))
            except:
                model = get_fasttext_model(config, train_df, save=True)
        else:
            model = get_fasttext_model(config, train_df, save=False)


    TRAIN_LENGTH, VALID_LENGTH = train_df.shape[0], valid_df.shape[0]

    df = pd.concat([train_df, valid_df, test_df], axis=0)
    breakfast = df.조식메뉴.values
    lunch = df.중식메뉴.values
    dinner = df.석식메뉴.values

    breakfast_array = np.zeros((1255, config.dim))
    lunch_array = np.zeros((1255, config.dim))
    dinner_array = np.zeros((1255, config.dim))
    for i in range(1255) :
        breakfast_array[i] = model.get_sentence_vector(breakfast[i]) / len(breakfast[i].split())
        lunch_array[i] = model.get_sentence_vector(lunch[i]) / len(lunch[i].split())
        dinner_array[i] = model.get_sentence_vector(dinner[i]) / len(dinner[i].split())

    for i in range(config.dim) :
        embedding_features.append('breakfast_{}'.format(i))
        embedding_features.append('lunch_{}'.format(i))
        embedding_features.append('dinner_{}'.format(i))

    tmp = pd.concat([
        pd.DataFrame(breakfast_array, columns=['breakfast_{}'.format(i) for i in range(config.dim)]),
        pd.DataFrame(lunch_array, columns=['lunch_{}'.format(i) for i in range(config.dim)]),
        pd.DataFrame(dinner_array, columns=['dinner_{}'.format(i) for i in range(config.dim)])], axis=1)
    train_tmp = tmp[:TRAIN_LENGTH]
    valid_tmp = tmp[TRAIN_LENGTH: TRAIN_LENGTH + VALID_LENGTH]
    test_tmp = tmp[TRAIN_LENGTH + VALID_LENGTH:]

    return train_tmp, valid_tmp, test_tmp

class Encoder(tf.keras.models.Model) :
    def __init__(self, step, input_size) :
        super(Encoder, self).__init__()
        self.model = tf.keras.models.Sequential([
            InputLayer(input_shape=(input_size,)),
            Dense(input_size - step * 1, activation='relu'),
            Dense(input_size - step * 2, activation='relu'),
            Dense(input_size - step * 3, activation='relu'),
            Dense(input_size - step * 4, activation='relu'),
            Dense(input_size - step * 5),
        ])

    def call(self, x) :
        z = self.model(x)
        return z

class Decoder(tf.keras.models.Model) :
    def __init__(self, step, input_size, output_size) :
        super(Decoder, self).__init__()
        self.model = tf.keras.models.Sequential([
            InputLayer(input_shape=(input_size,)),
            Dense(output_size - step * 4, activation='relu'),
            #             Dense(output_size - step * 3, activation='relu'),
            Dense(output_size - step * 2, activation='relu'),
            #             Dense(output_size - step * 1, activation='relu'),
            Dense(output_size),
        ])

    def call(self, x) :
        z = self.model(x)
        return z

class AutoEncoder(tf.keras.models.Model) :
    def __init__(self, input_size, step) :
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(step, input_size)
        self.decoder = Decoder(step, input_size - step * 5, input_size)

    def call(self, x) :
        y = self.encoder(x)
        z = self.decoder(y)
        return z

def autoencoding(config, train_df, valid_df, test_df) :
    TRAIN_LENGTH = train_df.shape[0]
    VALID_LENGTH = valid_df.shape[0]

    df = pd.concat([train_df, valid_df, test_df], axis=0)
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
    lunch = vect.transform(text_df['중식메뉴'].values).toarray()
    vect.fit(text_df['석식메뉴'].values[:TRAIN_LENGTH])
    dinner = vect.transform(text_df['석식메뉴'].values).toarray()

    enc_df = pd.DataFrame()

    i = 1
    for menu in [breakfast, lunch, dinner] :
        print('+' * 10, 'Train {}'.format(i), '+' * 10)
        train_X, valid_X = menu[:TRAIN_LENGTH], menu[TRAIN_LENGTH :TRAIN_LENGTH + VALID_LENGTH]
        STEP = (menu.shape[1] - config.dim) // 5
        INPUT = menu.shape[1]

        model = AutoEncoder(INPUT, STEP)
        model.compile(loss='mse', optimizer='adam', metrics='mse')
        model.fit(train_X, train_X,
                  validation_data=(valid_X, valid_X),
                  epochs=1000,
                  callbacks=tf.keras.callbacks.EarlyStopping(patience=10),
                  verbose=2)
        result = model.encoder(menu)
        result = np.array(result)

        enc_df = pd.concat([enc_df, pd.DataFrame(result)], axis=1)
        i += 1
        print()


    train_df_tmp = enc_df.iloc[:TRAIN_LENGTH]
    valid_df_tmp = enc_df.iloc[TRAIN_LENGTH :TRAIN_LENGTH + VALID_LENGTH]
    test_df_tmp = enc_df.iloc[TRAIN_LENGTH + VALID_LENGTH:]

    column_names = [i for i in range(enc_df.shape[1])]
    train_df_tmp.columns = column_names
    valid_df_tmp.columns = column_names
    test_df_tmp.columns = column_names

    return train_df_tmp, valid_df_tmp, test_df_tmp
