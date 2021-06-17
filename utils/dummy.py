import argparse
import copy
import random
import numpy as np


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--load_fn', required=True,
                   help='txt filename to be augmented')
    p.add_argument('--save_fn', required=True,
                   help='filename for result file (txt format)')
    p.add_argument('--iter', type=int, default=2,
                   help='the bigger this number, the larger number of data will be added as dummy')
    p.add_argument('--verbose', type=int, default=300,
                   help='verbosity')

    config = p.parse_args()

    return config

def get_corpus(config):
    with open(config.load_fn, 'r', -1, encoding='utf-8') as f:
        data = f.readlines()
    corpus = np.asarray([d.rstrip() for d in data if d != '']).reshape(-1, 1)
    return corpus


def dummy_data(config, corpus) :
    for i in range(corpus.shape[0]) :
        if i % config.verbose == 0:
            print('Train completed on {} datas'.format(i))

        string = np.array(corpus[i][0].split())
        new_string = np.random.permutation(string)
        corpus = np.concatenate([corpus, np.array(' '.join(new_string)).reshape(-1, 1)], axis=0)
        # (sentences, 1)
    
    # set으로 hashing을 위해선 1차원이어야 함
    # (sentences, )
    return np.array(list(set(corpus.reshape(-1))))

def main(config):
    corpus = get_corpus(config)
    new_corpus = copy.deepcopy(corpus)

    prev_shape = corpus.shape[0]
    if config.iter > 1:
        for k in range(config.iter) :
            print('|ITERATION {}|'.format(k + 1))
            new_corpus = np.concatenate([new_corpus, dummy_data(config, corpus).reshape(-1,1)], axis=0)
    else:
        new_corpus = dummy_data(config, corpus)
    new_corpus = np.array(list(set(new_corpus.reshape(-1))))

    post_shape = new_corpus.shape[0]

    print('COMPLETED: {} data is augmented'.format(post_shape - prev_shape))
    with open(config.save_fn, 'w', -1, encoding='utf-8') as f :
        f.write('\n'.join(new_corpus))

if __name__ == "__main__":
    config = define_argparser()
    main(config)