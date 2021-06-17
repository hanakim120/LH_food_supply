import argparse

import pandas as pd

from module.np_trainer import train_np
from module.lgbm_trainer import train_lgbm
from module.catboost_trainer import train_catboost
from module.data_loader import get_data
from utils.prophet_tools import save_np_submission

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, required=True,
                   help='model selection. result file will be saved as "submission_[model].csv"')
    # 'embedding' or 'tokenize' or 'raw'
    p.add_argument('--text', type=str, default='embedding',
                   help='Method to preprocess text data (embedding / tokenize / autoencode) ')
    p.add_argument('--sep_date', action='store_true',
                   help='When True, it does not use time column as timestamp. '
                        'it separates the columns with month and week and traet as categorical features')
    p.add_argument('--dummy_corpus', action='store_true',
                   help='making dummy data using random permutation')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')


    # fasttext
    p.add_argument('--fasttext_model_fn', type=str,
                   help='fasttext model file path (train new model if path invalid)')
    p.add_argument('--pretrained', action='store_true',
                   help='Use pretrained korean language model')
    p.add_argument('--use_tok', action='store_true',
                   help='use mecab tokenizing when embedding text data (Avalibable only if mecab installed on your environment')

    p.add_argument('--dim', type=int, default=3,
                   help='embedding dimension (only woking when using embedding or autoencoding method')
    p.add_argument('--min_count', type=int, default=0,
                   help='argument for fasttext')
    p.add_argument('--window_size', type=int, default=3,
                   help='argument for fasttext')
    p.add_argument('--min_ngram', type=int, default=2,
                   help='argument for fasttext')
    p.add_argument('--max_ngram', type=int, default=4,
                   help='argument for fasttext')
    p.add_argument('--fasttext_epoch', type=int, default=1000,
                   help='number of training epochs')

    # catboost
    p.add_argument('--depth', type=int, default=3,
                   help='growth stop criterion')
    p.add_argument('--verbose', type=int, default=100,
                   help='verbosity')
    p.add_argument('--k', type=int, default=0,
                   help='using k-fold when k > 0')
    p.add_argument('--has_time', type=bool, default=True,
                   help='whether randomly shuffle datas or not')

    # lgbm
    p.add_argument('--num_leaves', type=int, default=3,
                   help='number of leaves limitation for growth stopping')
    p.add_argument('--min_data_in_leaf', type=int, default=5,
                   help='minimum number of data in leaf when the tree growing')
    p.add_argument('--min_sum_hessian_in_leaf', type=int, default=5,
                   help='minimum sum of hessian(gradient of gradient)')
    p.add_argument('--max_bin', type=int, default=180,
                   help='maximum number of bin for features')
    p.add_argument('--min_data_in_bin', type=int, default=3,
                   help='minimum data in bin for features')
    p.add_argument('--max_cat_threshold', type=int, default=2,
                   help='maximum threshold for categorical feature treatment')
    p.add_argument('--lr', type=float, default=0.04,
                   help='learning rate')


    # experimental
    p.add_argument('--sum_reduction', action='store_true',
                   help='sum all embedding values and divide into embedding dimension')
    p.add_argument('--drop_text', action='store_true',
                   help='drop text columns (set methods to "embedding")')
    config = p.parse_args()

    return config

def get_model(config, train_df, valid_df, train_y, valid_y):
    if config.model == 'catboost':
        reg_launch, reg_dinner = train_catboost(train_df,
                                                valid_df,
                                                train_y,
                                                valid_y,
                                                config)
    elif config.model == 'lgbm':
        reg_launch, reg_dinner = train_lgbm(train_df,
                                            valid_df,
                                            train_y,
                                            valid_y,
                                            config)

    elif config.model == 'np':
        reg_launch, reg_dinner = train_np(config, train_df, y)

    return reg_launch, reg_dinner

def save_submission(config, reg_launch, reg_dinner, test_df, sample, file_fn='new'):
    if type(reg_launch) == dict:
        k = config.k
        for i in range(k):
            sample.중식계 += reg_launch[i].predict(test_df)
            sample.석식계 += reg_dinner[i].predict(test_df)

        sample.중식계 /= k
        sample.석식계 /= k
        sample.to_csv('./result/submission_{}.csv'.format(file_fn), index=False)
    else:
        sample.중식계 = reg_launch.predict(test_df)
        sample.석식계 = reg_dinner.predict(test_df)

        sample.to_csv('./result/submission_{}.csv'.format(file_fn), index=False)
    print('=' * 10, 'SAVE COMPLETED', '=' * 10)
    print(sample.tail())


def main(config):
    train_df, valid_df, test_df, train_y, valid_y, sample = get_data(config)

    print(train_df.head())
    if config.drop_text:
        train_df = train_df.iloc[:, :10]
        test_df = test_df.iloc[:, :10]

    if config.model == 'np':
        train_df = pd.concat([train_df['ds'], train_y], axis=1)
        valid_df = pd.concat([valid_df['ds'], valid_y], axis=1)
        test_df = test_df[['ds']]

    reg_launch, reg_dinner = get_model(config, train_df, valid_df, train_y, valid_y)

    return reg_launch, reg_dinner, train_df, valid_df, test_df, sample

if __name__ == '__main__':
    config = define_argparser()
    if config.model == 'np' :
        reg_launch, reg_dinner, train_df, valid_df, test_df, sample = main(config)
        save_np_submission(config, train_df, reg_launch, reg_dinner, sample)
    else:
        reg_launch, reg_dinner, _, valid_df, test_df, sample = main(config)
        save_submission(config, reg_launch, reg_dinner, test_df, sample, file_fn=config.model)



