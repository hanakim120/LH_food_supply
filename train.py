import argparse
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd

from module.data_loader import get_data
from module.model.lgbm import train_lgbm
from module.model.catboost import train_catboost
from module.model.tabnet import train_tabnet
from module.model.tree import train_randomforest, train_extratree, train_gradient_boosting, train_decision_tree
from module.model.regularized_regression import train_reg_model
from module.model.linear_regression import train_lr

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, required=True,
                   help='''model selection. result file will be saved as "./data/submission_[model].csv"
                   (catboost / lgbm / lr / rf / reg / tabnet)''')

    p.add_argument('--text', type=str, default='menu',
                   help='Method to preprocess text data (embedding / menu / autoencode)')
    p.add_argument('--train_size', type=int, default=1000)
    p.add_argument('--menu_fn', type=str, default='embedding.oov',
                   help='menu embedding file name (csv format)')
    p.add_argument('--oov_cnt', action='store_true')
    p.add_argument('--holiday_length', action='store_true')
    p.add_argument('--dim', type=int, default=3,
                   help='embedding dimension')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--dummy_cat', action='store_true',
                   help='transform categorical feateatures into dummy variables')
    p.add_argument('--k', type=int, default=0,
                   help='using k-fold when k > 0')
    p.add_argument('--verbose', type=int, default=100,
                   help='verbosity')
    p.add_argument('--epochs', type=int, default=1000,
                   help='epochs')


    # fasttext
    p.add_argument('--fasttext_model_fn', type=str, default='basemodel.bin',
                   help='fasttext model file path (train new model if path invalid)')
    p.add_argument('--pretrained', action='store_true',
                   help='Use pretrained korean language model')
    p.add_argument('--use_tok', action='store_true',
                   help='use mecab tokenizing when embedding text data (Avalibable only if mecab installed on your environment')
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
    p.add_argument('--dummy_corpus', action='store_true',
                   help='making dummy data using random permutation')

    # catboost
    p.add_argument('--depth', type=int, default=3,
                   help='growth stop criterion')
    p.add_argument('--has_time', type=bool, default=True,
                   help='whether randomly shuffle datas or not')
    p.add_argument('--rsm', type=float, default=1.0,
                   help='random feature sampling ratio')

    # Tabnet
    p.add_argument('--n_d', type=int, default=8,
                   help='''Width of the decision prediction layer. 
                   Bigger values gives more capacity to the model with the risk of overfitting. 
                   Values typically range from 8 to 64.
                   (n_d == n_a)''')
    p.add_argument('--n_steps', type=int, default=3,
                   help='Number of steps in the architecture (usually between 3 and 10)')
    p.add_argument('--gamma', type=float, default=1.3,
                   help='''This is the coefficient for feature reusage in the masks.
                    A value close to 1 will make mask selection least correlated between layers. 
                    Values range from 1.0 to 2.0.''')
    p.add_argument('--lambda_sparse', type=float, default=1e-3,
                   help='''This is the extra sparsity loss coefficient as proposed in the original paper. 
                   The bigger this coefficient is, the sparser your model will be in terms of feature selection. 
                   Depending on the difficulty of your problem, reducing this value could help.''')
    p.add_argument('--n_independent', type=int, default=1,
                   help='Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.')
    p.add_argument('--n_shared', type=int, default=2,
                   help='Number of shared Gated Linear Units at each step Usual values range from 1 to 5')
    p.add_argument('--use_radam', action='store_true',
                   help='use radam as optimizer for training tabnet, oterwise use SGD instead.')
    p.add_argument('--max_grad_norm', type=float, default=1e8,
                   help='gradient clipping')
    p.add_argument('--momentum', type=float, default=0.02,
                   help='Momentum for batch normalization, typically ranges from 0.01 to 0.4 (default=0.02)')
    p.add_argument('--batch_size', type=int, default=200,
                   help='batch size')
    p.add_argument('--lr_decay_start', type=int, default=120,
                   help='when to start learning rate decay')
    p.add_argument('--lr_step', type=int, default=30,
                   help='apply learning rate decay with every N step')
    p.add_argument('--lr_gamma', type=float, default=0.9,
                   help='learning rate reduction ratio')
    p.add_argument('--optim_lr', type=float, default=1e-1,
                   help='initialize learning rate for optimizer')

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

    # rf
    p.add_argument('--n_est', type=int, default=200,
                   help='number_of_estimators')
    p.add_argument('--min_samp', type=int, default=1,
                   help='minimum number of sample to split node')

    # reg
    p.add_argument('--alpha', type=float, default=1.)

    # experimental
    p.add_argument('--sum_reduction', action='store_true',
                   help='sum all embedding values and divide into embedding dimension')
    p.add_argument('--drop_text', action='store_true',
                   help='drop text columns')
    config = p.parse_args()

    return config

def get_model(
        config,
        train_df,
        valid_df,
        train_y,
        valid_y,
        lunch_drop_col,
        dinner_drop_col
):
    if config.model == 'catboost':
        reg_lunch, reg_dinner = train_catboost(train_df,
                                                valid_df,
                                                train_y,
                                                valid_y,
                                                config)
    elif config.model == 'lgbm':
        reg_lunch, reg_dinner = train_lgbm(train_df,
                                            valid_df,
                                            train_y,
                                            valid_y,
                                            config)

    elif config.model == 'tabnet':
        reg_lunch, reg_dinner = train_tabnet(train_df,
                                             valid_df,
                                             train_y,
                                             valid_y,
                                             config)
    elif config.model == 'rf':
        reg_lunch, reg_dinner = train_randomforest(train_df,
                                                   valid_df,
                                                   train_y,
                                                   valid_y,
                                                   config)
    elif config.model == 'et':
        reg_lunch, reg_dinner = train_extratree(train_df,
                                                   valid_df,
                                                   train_y,
                                                   valid_y,
                                                   config)
    elif config.model == 'gb':
        reg_lunch, reg_dinner = train_gradient_boosting(train_df,
                                                   valid_df,
                                                   train_y,
                                                   valid_y,
                                                   config)
    elif config.model == 'reg':
        reg_lunch, reg_dinner = train_reg_model(train_df,
                                                valid_df,
                                                train_y,
                                                valid_y,
                                                config)
    elif config.model == 'lr':
        reg_lunch, reg_dinner = train_lr(train_df,
                                         valid_df,
                                         train_y,
                                         valid_y,
                                         config,
                                         lunch_drop_col,
                                         dinner_drop_col)
    elif config.model == 'dt':
        reg_lunch, reg_dinner = train_decision_tree(train_df,
                                         valid_df,
                                         train_y,
                                         valid_y,
                                         config)
    else:
        print('ERROR: INVALID MODEL NAME (available: catboost / lgbm / tabnet / rf / reg/ lr)')
        quit()

    return reg_lunch, reg_dinner

def save_submission(config, reg_lunch, reg_dinner, test_df, sample, lunch_drop_col, dinner_drop_col, file_fn='new'):
    test_lunch = test_df.drop(columns=lunch_drop_col)
    test_dinner = test_df.drop(columns=dinner_drop_col)
    if type(reg_lunch) == dict:
        k = config.k
        for i in range(k):
            sample.중식계 += reg_lunch[i].predict(test_lunch)
            sample.석식계 += reg_dinner[i].predict(test_dinner)

        sample.중식계 /= k
        sample.석식계 /= k
        sample.to_csv('./result/submission_{}.csv'.format(file_fn), index=False)
    else:
        if config.model == 'tabnet':
            sample.중식계 = reg_lunch.predict(test_lunch.values)
            sample.석식계 = reg_dinner.predict(test_dinner.values)
        else:
            sample.중식계 = reg_lunch.predict(test_lunch)
            sample.석식계 = reg_dinner.predict(test_dinner)

        sample.to_csv('./result/submission_{}.csv'.format(file_fn), index=False)

    base = pd.read_csv('./result/submission_lr.base.csv', encoding='utf-8')

    lunch_diff = 0
    dinner_diff = 0
    for i in range(50) :
        lunch_diff += abs(base.중식계[i] - sample.중식계[i])
        dinner_diff += abs(base.석식계[i] - sample.석식계[i])
    lunch_diff /= 50
    dinner_diff /= 50
    total_diff = (lunch_diff + dinner_diff) / 2

    print('DIFFRENCE (LUNCH) :', lunch_diff if lunch_diff > 1e-3 else 0)
    print('DIFFRENCE (DINNER) :', dinner_diff if dinner_diff > 1e-3 else 0)
    print('AVERAGE DIFFERENCE :', total_diff if total_diff > 1e-3 else 0)
    print('=' * 10, 'SAVE COMPLETED', '=' * 10)
    print(sample.head())


def main(config):
    train_df, valid_df, test_df, train_y, valid_y, sample = get_data(config)

    lunch_drop_col = ['공휴일길이']
    dinner_drop_col = ['강수량',
                       'breakfast_0', 'breakfast_1', 'breakfast_2',
                       'lunch_0', 'lunch_1', 'lunch_2',
                       'dinner_0', 'dinner_1', 'dinner_2']

    reg_lunch, reg_dinner = get_model(
        config,
        train_df,
        valid_df,
        train_y,
        valid_y,
        lunch_drop_col,
        dinner_drop_col
    )
    save_submission(config, reg_lunch, reg_dinner, test_df, sample, lunch_drop_col, dinner_drop_col, file_fn=config.model)


if __name__ == '__main__':
    config = define_argparser()
    main(config)




