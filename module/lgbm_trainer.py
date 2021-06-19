import random

import numpy as np
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


def train_lgbm(train_df,
               valid_df,
               train_y,
               valid_y,
               config):

    cat_features = [0, 2, 3, 4, 13, 15] # 요일, 공휴일전후_0, 공휴일전후_1, 공휴일전후_2, month, week

    if config.k > 0:
        kf = KFold(n_splits=config.k, shuffle=True, random_state=42)
        folds = []
        for train_idx, valid_idx in kf.split(train_df, train_y.중식계) :
            folds.append((train_idx, valid_idx))

        random.seed(config.seed)
        reg_launch = {}
        reg_dinner = {}
        scores_launch = []
        scores_dinner = []
        for fold in range(config.k) :
            print(f'===================================={fold + 1}============================================')
            train_idx, valid_idx = folds[fold]
            X_train, X_valid, y_train, y_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx], \
                                                 train_y.중식계.iloc[train_idx], train_y.중식계.iloc[valid_idx]

            reg_launch_ = LGBMRegressor(
                objective='regression_l2',
                learning_rate=config.lr,
                num_leaves=config.num_leaves,
                max_depth=config.depth,
                min_data_in_leaf=config.min_data_in_leaf,
                min_sum_hessian_in_leaf=config.min_sum_hessian_in_leaf,
                max_bin=config.max_bin,
                min_data_in_bin=config.min_data_in_bin,
                max_cat_threshold=config.max_cat_threshold,
                metric='mae',
                num_iterations=3000,
                early_stopping_rounds=100,
                random_state=config.seed,
            )

            print('=' * 10, 'TRAIN LAUNCH MODEL', '=' * 10)
            reg_launch_.fit(X_train, y_train,
                           eval_set=[(X_train, y_train), (X_valid, y_valid)],
                           eval_names=['train', 'valid'],
                           eval_metric='mae',
                           categorical_feature=cat_features,
                           verbose=config.verbose)

            reg_launch[fold] = reg_launch_
            y_pred = reg_launch_.predict(valid_df)
            mae_launch = mean_absolute_error(valid_y.중식계, y_pred)
            print('VALID SCORE: ', mae_launch)
            scores_launch.append(mae_launch)

            # =====================================================
            y_train, y_valid = train_y.석식계.iloc[train_idx], train_y.석식계.iloc[valid_idx]

            reg_dinner_ = LGBMRegressor(
                objective='regression_l2',
                learning_rate=config.lr,
                num_leaves=config.num_leaves,
                max_depth=config.depth,
                min_data_in_leaf=config.min_data_in_leaf,
                min_sum_hessian_in_leaf=config.min_sum_hessian_in_leaf,
                max_bin=config.max_bin,
                min_data_in_bin=config.min_data_in_bin,
                max_cat_threshold=config.max_cat_threshold,
                metric='mae',
                num_iterations=3000,
                early_stopping_rounds=100,
                random_state=config.seed,
            )
            print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)
            reg_dinner_.fit(X_train, y_train,
                           eval_set=[(X_train, y_train), (X_valid, y_valid)],
                           eval_names=['train', 'valid'],
                           eval_metric='mae',
                           categorical_feature=cat_features,
                           verbose=config.verbose)

            reg_dinner[fold] = reg_dinner_
            y_pred = reg_dinner_.predict(valid_df)
            mae_dinner = mean_absolute_error(valid_y.석식계, y_pred)
            print('VALID SCORE: ', mae_dinner)
            scores_dinner.append(mae_dinner)
            print(f'================================================================================\n\n')

        print('-' * 50, 'Result', '-' * 50, '\n')
        print('Mean launch : {}'.format(np.mean(scores_launch, axis=0)))
        print('Mean dinner : {}'.format(np.mean(scores_dinner, axis=0)))
        print('Total : {}'.format((np.mean(scores_launch, axis=0) + np.mean(scores_dinner, axis=0)) / 2))
        print('_' * 106)

        return reg_launch, reg_dinner
    else:
        reg_launch = LGBMRegressor(
            objective='regression_l2',
            learning_rate=config.lr,
            num_leaves=config.num_leaves,
            max_depth=config.depth,
            min_data_in_leaf=config.min_data_in_leaf,
            min_sum_hessian_in_leaf=config.min_sum_hessian_in_leaf,
            max_bin=config.max_bin,
            min_data_in_bin=config.min_data_in_bin,
            max_cat_threshold=config.max_cat_threshold,
            metric='mae',
            num_iterations=3000,
            early_stopping_rounds=100,
            random_state=config.seed,
        )
        print('=' * 10, 'TRAIN LAUNCH MODEL', '=' * 10)
        reg_launch.fit(train_df, train_y.중식계,
                 eval_set=[(train_df, train_y.중식계), (valid_df, valid_y.중식계)],
                 eval_names=['train', 'valid'],
                 eval_metric='mae',
                 categorical_feature=cat_features,
                 verbose=config.verbose)

        mae_launch = reg_launch.best_score_['valid']['l1']
        print('BEST SCORE: ', mae_launch)
        print()

        reg_dinner = LGBMRegressor(
            objective='regression_l2',
            learning_rate=config.lr,
            num_leaves=config.num_leaves,
            max_depth=config.depth,
            min_data_in_leaf=config.min_data_in_leaf,
            min_sum_hessian_in_leaf=config.min_sum_hessian_in_leaf,
            max_bin=config.max_bin,
            min_data_in_bin=config.min_data_in_bin,
            max_cat_threshold=config.max_cat_threshold,
            metric='mae',
            num_iterations=3000,
            early_stopping_rounds=100,
            random_state=config.seed,
        )
        print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)
        reg_dinner.fit(train_df, train_y.석식계,
                 eval_set=[(train_df, train_y.석식계), (valid_df, valid_y.석식계)],
                 eval_names=['train', 'valid'],
                 eval_metric='mae',
                 categorical_feature=cat_features,
                 verbose=config.verbose)

        mae_dinner = reg_dinner.best_score_['valid']['l1']
        print('BEST SCORE: ', mae_dinner)
        print()

        print('=' * 10, 'RESULT', '=' * 10)
        print('MEAN :', (mae_dinner + mae_launch) / 2 )

        return reg_launch, reg_dinner