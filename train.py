import argparse
import warnings
warnings.filterwarnings(action='ignore')

from module.lgbm_trainer import train_lgbm
from module.catboost_trainer import train_catboost
from module.tabnet_trainer import train_tabnet
from module.data_loader import get_data

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, required=True,
                   help='model selection. result file will be saved as "submission_[model].csv"')
    # 'embedding' or 'tokenize' or 'raw'
    p.add_argument('--text', type=str, default='embedding',
                   help='Method to preprocess text data (embedding / tokenize / autoencode) ')
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
                   help='use radam as optimizer for training tabnet, oterwise use adam instead.')
    p.add_argument('--epochs', type=int, default=1000,
                   help='epochs')
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


    # experimental
    p.add_argument('--sum_reduction', action='store_true',
                   help='sum all embedding values and divide into embedding dimension')
    p.add_argument('--drop_text', action='store_true',
                   help='drop text columns (set methods to "embedding")')
    config = p.parse_args()

    return config

def get_model(config, train_df, valid_df, train_y, valid_y):
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
    else:
        print('ERROR: INVALID MODEL NAME (available: catboost / lgbm / tabnet)')
        quit()

    return reg_lunch, reg_dinner

def save_submission(config, reg_lunch, reg_dinner, test_df, sample, file_fn='new'):
    if type(reg_lunch) == dict:
        k = config.k
        for i in range(k):
            sample.중식계 += reg_lunch[i].predict(test_df)
            sample.석식계 += reg_dinner[i].predict(test_df)

        sample.중식계 /= k
        sample.석식계 /= k
        sample.to_csv('./result/submission_{}.csv'.format(file_fn), index=False)
    else:
        sample.중식계 = reg_lunch.predict(test_df.values)
        sample.석식계 = reg_dinner.predict(test_df.values)

        sample.to_csv('./result/submission_{}.csv'.format(file_fn), index=False)
    print('=' * 10, 'SAVE COMPLETED', '=' * 10)
    print(sample.head())


def main(config):
    train_df, valid_df, test_df, train_y, valid_y, sample = get_data(config)

    if config.drop_text:
        train_df = train_df.iloc[:, : len(train_df.columns) - 3 * config.dim]
        test_df = test_df.iloc[:, : len(train_df.columns) - 3 * config.dim]

    reg_lunch, reg_dinner = get_model(config, train_df, valid_df, train_y, valid_y)

    return reg_lunch, reg_dinner, train_df, valid_df, test_df, sample

if __name__ == '__main__':
    config = define_argparser()
    reg_lunch, reg_dinner, _, valid_df, test_df, sample = main(config)
    save_submission(config, reg_lunch, reg_dinner, test_df, sample, file_fn=config.model)



