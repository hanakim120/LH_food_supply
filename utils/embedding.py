import re

import numpy as np
import pandas as pd

import fasttext
from konlpy.tag import Mecab

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer


TRAIN_LENGTH = 1205


def pretext(text) :
    text = text.rstrip()
    text = re.sub('▁', '', text)
    return text

def embedding(config, breakfast, launch, dinner):
    embedding_features = []

    if config.pretrained:
        print('USE PRETRAINED MODEL: cc.ko.300.bin')
        model = fasttext.load_model('cc.ko.300.bin')
        fasttext.util.reduce_model(model, config.dim)

    else:
        try:
            model = fasttext.load_model('./data/{}'.format(config.fasttext_model_fn))
        except:
            print('"{}" DOES NOT EXIST. START TRAINING MODEL'.format(config.model))
            model = fasttext.train_unsupervised('./data/corpus.train.dummy.txt' if config.dummy_corpus else './data/corpus.train.txt',
                                                dim=config.dim,
                                                ws=config.window_size,
                                                epoch=config.fasttext_epoch,
                                                min_count=config.min_count,
                                                minn=config.min_ngram,
                                                maxn=config.max_ngram,
                                                )
            model.save_model('./data/{}'.format(config.fasttext_model_fn))

    breakfast_array = np.zeros((1255, config.dim))
    launch_array = np.zeros((1255, config.dim))
    dinner_array = np.zeros((1255, config.dim))

    mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

    for i in range(1255) :
        if config.use_tok:
            breakfast[i] = ' '.join(mecab.morphs(breakfast[i]))
            launch[i] = ' '.join(mecab.morphs(launch[i]))
            dinner[i] = ' '.join(mecab.morphs(dinner[i]))

        breakfast_array[i] = model.get_sentence_vector(breakfast[i]) / (len(breakfast[i].split()) + 1)
        launch_array[i] = model.get_sentence_vector(launch[i]) / (len(launch[i].split()) + 1)
        dinner_array[i] = model.get_sentence_vector(dinner[i]) / (len(dinner[i].split()) + 1)

    for i in range(config.dim) :
        embedding_features.append('breakfast_{}'.format(i))
        embedding_features.append('launch_{}'.format(i))
        embedding_features.append('dinner_{}'.format(i))

    tmp = pd.concat([
        pd.DataFrame(breakfast_array, columns=['breakfast_{}'.format(i) for i in range(config.dim)]),
        pd.DataFrame(launch_array, columns=['launch_{}'.format(i) for i in range(config.dim)]),
        pd.DataFrame(dinner_array, columns=['dinner_{}'.format(i) for i in range(config.dim)])], axis=1)

    return tmp

def subword(config):
    # corpus = '\n'.join(np.concatenate([breakfast[:TRAIN_LENGTH],
    #                                    launch[:TRAIN_LENGTH],
    #                                    dinner[:TRAIN_LENGTH]], axis=0))
    # with open('./data/corpus.train.txt', 'w', encoding='utf-8') as f:
    #     f.write(corpus)
    #
    # corpus = '\n'.join(np.concatenate([breakfast[TRAIN_LENGTH:],
    #                                    launch[TRAIN_LENGTH:],
    #                                    dinner[TRAIN_LENGTH:]], axis=0))
    # with open('./data/corpus.test.txt', 'w', encoding='utf-8') as f:
    #     f.write(corpus)

    # =================== subword segment=========================
    with open('./data/corpus.train.sub.txt', 'r', encoding='utf-8') as f :
        data = f.readlines()
    corpus_sub_train = list(map(pretext, data))
    with open('./data/corpus.valid.sub.txt', 'r', encoding='utf-8') as f :
        data = f.readlines()
    corpus_sub_valid = list(map(pretext, data))
    with open('./data/corpus.test.sub.txt', 'r', encoding='utf-8') as f :
        data = f.readlines()
    corpus_sub_test = list(map(pretext, data))

    tokenizer = Tokenizer(oov_token='<oov>')
    tokenizer.fit_on_texts(corpus_sub_train)

    vocab_size = len(tokenizer.word_index)
    print('VOCAB SIZE : {}'.format(vocab_size))
    cnt = 0
    for i in tokenizer.word_counts.values():
        if i == 1:
            cnt += 1
    print('Freq 1 word : {}'.format(cnt))
    tokenizer = Tokenizer(oov_token='<oov>', num_words=vocab_size \
        if not config.sub_sparse_word \
        else vocab_size - cnt + 2)

    tokenizer.fit_on_texts(corpus_sub_train)

    corpus_sub = corpus_sub_train + corpus_sub_valid + corpus_sub_test
    corpus_seq = tokenizer.texts_to_sequences(corpus_sub)

    embeds = nn.Embedding(839, config.dim)

    def get_sentence_vect(seq) :
        result = torch.tensor(torch.zeros((config.dim)))
        for idx in seq :
            result += embeds(torch.tensor(idx, dtype=torch.long))
        return (result / len(seq)).detach().numpy()

    tmp = pd.DataFrame(np.array(list(map(get_sentence_vect, corpus_seq))))
    tmp = pd.concat([tmp.iloc[:1255, :],
                     tmp.iloc[1255 :2510, :].reset_index(drop=True),
                     tmp.iloc[2510 :, :].reset_index(drop=True)], axis=1)
    columns = []

    for menu in ['breakfast', 'launch', 'dinner']:
        for i in range(config.dim):
            columns.append('{}_{}'.format(menu, i))

    tmp.columns = columns

    return tmp

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