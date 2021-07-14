import re
from argparse import Namespace

import numpy as np
import pandas as pd
from tqdm import tqdm

import fasttext
from konlpy.tag import Mecab

import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

from module.model.autoencoder import AutoEncoder
from utils.dummy import main


TRAIN_LENGTH = 1205


def pretext(text) :
    text = text.rstrip()
    text = re.sub('▁', '', text)
    return text

def get_fasttext_model(config, train_df, valid_df, save=False):
    print('"{}" DOES NOT EXIST. START TRAINING MODEL'.format(config.fasttext_model_fn))

    corpus = np.concatenate([train_df.조식메뉴.values,
                             train_df.중식메뉴.values,
                             train_df.석식메뉴.values], axis=0)
    with open('./data/fasttext/corpus.train.txt', 'w', -1, encoding='utf-8') as f :
        f.write('\n'.join(corpus))

    if config.dummy_corpus :
        args = {'load_fn' : './data/fasttext/corpus.train.txt',
                'save_fn' : './data/fasttext/corpus.train.dummy.txt',
                'iter' : 10,
                'verbose' : 300}
        args = Namespace(**args)
        main(args)

    model = fasttext.train_unsupervised(
        './data/fasttext/corpus.train.dummy.txt' if config.dummy_corpus else './data/fasttext/corpus.train.txt',
        dim=config.dim,
        ws=config.window_size,
        epoch=config.fasttext_epoch,
        min_count=config.min_count,
        minn=config.min_ngram,
        maxn=config.max_ngram,
        )
    if save:
        model.save_model('./data/fasttext/{}'.format(config.fasttext_model_fn))
    return model

def embedding(config, train_df, valid_df, test_df):
    if config.drop_text:
        train_tmp = pd.DataFrame()
        valid_tmp = pd.DataFrame()
        test_tmp = pd.DataFrame()
    else:
        if config.text == 'embedding':
            embedding_features = []

            if config.pretrained:
                print('USE PRETRAINED MODEL: cc.ko.300.bin')
                model = fasttext.load_model('cc.ko.300.bin')
                fasttext.util.reduce_model(model, config.dim)

            else:
                if config.fasttext_model_fn is not None:
                    try:
                        model = fasttext.load_model('./data/fasttext/{}'.format(config.fasttext_model_fn))
                    except:
                        model = get_fasttext_model(config, train_df, valid_df, save=True)
                else:
                    model = get_fasttext_model(config, train_df, valid_df, save=False)


            TRAIN_LENGTH, VALID_LENGTH = train_df.shape[0], valid_df.shape[0]

            df = pd.concat([train_df, valid_df, test_df], axis=0)
            breakfast = df.조식메뉴.values
            lunch = df.중식메뉴.values
            dinner = df.석식메뉴.values

            breakfast_array = np.zeros((1255, config.dim))
            lunch_array = np.zeros((1255, config.dim))
            dinner_array = np.zeros((1255, config.dim))
            for i in range(1255) :
                breakfast_array[i] = model.get_sentence_vector(breakfast[i]) / (len(breakfast[i].split()) + 1)
                lunch_array[i] = model.get_sentence_vector(lunch[i]) / (len(lunch[i].split()) + 1)
                dinner_array[i] = model.get_sentence_vector(dinner[i]) / (len(dinner[i].split()) + 1)

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

        #============================================================================
        elif config.text == 'autoencode':
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

            train_tmp = enc_df.iloc[:TRAIN_LENGTH]
            valid_tmp = enc_df.iloc[TRAIN_LENGTH :TRAIN_LENGTH + VALID_LENGTH]
            test_tmp = enc_df.iloc[TRAIN_LENGTH + VALID_LENGTH :]

            column_names = [i for i in range(enc_df.shape[1])]
            train_tmp.columns = column_names
            valid_tmp.columns = column_names
            test_tmp.columns = column_names

        # ============================================================================
        elif config.text == 'menu':
            try :
                result = pd.read_csv('./data/menu_embedding/{}.csv'.format(config.menu_fn), encoding='utf-8')
                print(result.tail())
            except :
                print('MENU EMBEDDING FILE DOES NOT EXIST. MAKING EMBEDDING DATA.')

                menu = pd.read_csv('./data/menu_embedding/menu.sub.csv', encoding='utf-8')
                menu_list = list(menu.menu)

                # implemented by bpe alrogithm
                with open('./data/menu_embedding/breakfast.sub.txt', 'r', -1, encoding='utf-8') as f :
                    b = f.readlines()

                with open('./data/menu_embedding/lunch.sub.txt', 'r', -1, encoding='utf-8') as f :
                    l = f.readlines()

                with open('./data/menu_embedding/dinner.sub.txt', 'r', -1, encoding='utf-8') as f :
                    d = f.readlines()

                print('EMBEDDING BREAKFAST')
                for i in range(len(b)) :
                    b[i] = b[i].rstrip()
                    b[i] = re.sub('▁', '', b[i])

                array = [d.split() for d in b]
                breakfast = pd.DataFrame(np.zeros((len(array), menu.shape[1])))
                for i in tqdm(range(len(array))) :
                    for j in range(len(array[i])) :
                        if array[i][j] in menu_list :
                            breakfast.iloc[i, :23] += list(map(int, menu[menu['menu'] == array[i][j]].values.reshape(-1)[1 :]))
                        else:
                            breakfast.iloc[i, 23] += 1
                breakfast.columns = ['breakfast_{}'.format(i) for i in range(23)] + ['oov_cnt.b']
                print(breakfast.tail())

                print('EMBEDDING LUNCH')
                for i in range(len(l)) :
                    l[i] = l[i].rstrip()
                    l[i] = re.sub('▁', '', l[i])

                array = [d.split() for d in l]
                lunch = pd.DataFrame(np.zeros((len(array), menu.shape[1])))
                for i in tqdm(range(len(array))):
                    for j in range(len(array[i])) :
                        if array[i][j] in menu_list :
                            lunch.iloc[i, :23] += list(map(int, menu[menu['menu'] == array[i][j]].values.reshape(-1)[1 :]))
                        else:
                            lunch.iloc[i, 23] += 1
                lunch.columns = ['lunch_{}'.format(i) for i in range(23)] + ['oov_cnt.l']
                print(lunch.tail())

                print('EMBEDDING DINNER')
                for i in range(len(d)) :
                    d[i] = d[i].rstrip()
                    d[i] = re.sub('▁', '', d[i])

                array = [x.split() for x in d]
                dinner = pd.DataFrame(np.zeros((len(array), menu.shape[1])))
                for i in tqdm(range(len(array))):
                    for j in range(len(array[i])) :
                        if array[i][j] in menu_list :
                            dinner.iloc[i, :23] += list(map(int, menu[menu['menu'] == array[i][j]].values.reshape(-1)[1 :]))
                        else:
                            dinner.iloc[i, 23] += 1
                dinner.columns = ['dinner_{}'.format(i) for i in range(23)] + ['oov_cnt.d']
                print(dinner.tail())

                result = pd.concat([breakfast.reset_index(drop=True),
                                    lunch.reset_index(drop=True),
                                    dinner.reset_index(drop=True)], axis=1)
                print(result.tail())
                result.to_csv('./data/menu_embedding/{}.csv'.format(config.menu_fn), index=False)

        if config.dim > 0 :
            from sklearn.decomposition import PCA

            if config.oov_cnt:
                oov_cnt_b = result['oov_cnt.b'].values.reshape(-1, 1)
                oov_cnt_d = result['oov_cnt.d'].values.reshape(-1, 1)
                oov_cnt_l = result['oov_cnt.l'].values.reshape(-1, 1)

            b_pca = PCA(n_components=config.dim)
            l_pca = PCA(n_components=config.dim)
            d_pca = PCA(n_components=config.dim)

            b_pca.fit(result.iloc[:train_df.shape[0] + valid_df.shape[0], :23])
            l_pca.fit(result.iloc[:train_df.shape[0] + valid_df.shape[0], 24 :47])
            d_pca.fit(result.iloc[:train_df.shape[0] + valid_df.shape[0], 48 :71])

            # print('========== PCA RESULT ==========')
            # print('breakfast explained variance', np.cumsum(b_pca.explained_variance_ratio_))
            # print('lunch explained variance', np.cumsum(l_pca.explained_variance_ratio_))
            # print('dinner explained variance', np.cumsum(d_pca.explained_variance_ratio_))
            # print()

            breakfast = b_pca.transform(result.iloc[:, :23].values)
            lunch = l_pca.transform(result.iloc[:, 24 :47].values)
            dinner = d_pca.transform(result.iloc[:, 48 :71].values)

            concat_list = [breakfast, lunch, dinner, oov_cnt_b, oov_cnt_l, oov_cnt_d] if config.oov_cnt else [breakfast, lunch, dinner]
            result = pd.DataFrame(np.concatenate(concat_list, axis=1))

            if config.oov_cnt:
                result.columns = ['breakfast_{}'.format(i) for i in range(config.dim)] + \
                                 ['lunch_{}'.format(i) for i in range(config.dim)] + \
                                 ['dinner_{}'.format(i) for i in range(config.dim)] + \
                                 ['oov_cnt.b', 'oov_cnt.l', 'oov_cnt.d']
            else:
                result.columns = ['breakfast_{}'.format(i) for i in range(config.dim)] + \
                                 ['lunch_{}'.format(i) for i in range(config.dim)] + \
                                 ['dinner_{}'.format(i) for i in range(config.dim)]

            train_tmp = result[:train_df.shape[0]]
            valid_tmp = result[train_df.shape[0] :train_df.shape[0] + valid_df.shape[0]]
            test_tmp = result[train_df.shape[0] + valid_df.shape[0] :]

        else:
            print('ERROR: INVALID EMBEDDING METHOD, RUNTIME TERMINATED')
            quit()

    return train_tmp, valid_tmp, test_tmp
