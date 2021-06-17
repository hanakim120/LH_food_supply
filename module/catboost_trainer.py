import random

import numpy as np

from catboost import CatBoostRegressor, Pool

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def train_catboost(train_df,
                   valid_df,
                   train_y,
                   valid_y,
                   config):

    cat_features = [0, 8, 9] if train_df.columns[8] == 'month' else []

    if config.k > 0 :
        skf = KFold(n_splits=config.k, shuffle=True, random_state=config.seed)
        folds = []
        for train_idx, valid_idx in skf.split(train_df, train_y.중식계) :
            folds.append((train_idx, valid_idx))

        random.seed(42)
        reg_launch = {}
        reg_dinner = {}
        scores_launch = []
        scores_dinner = []
        for fold in range(config.k) :
            print(f'===================================={fold + 1}============================================')
            train_idx, valid_idx = folds[fold]
            X_train, X_valid, y_train, y_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx], \
                                                 train_y.중식계[train_idx], train_y.중식계[valid_idx]

            train_pool_launch = Pool(X_train, y_train, cat_features=cat_features)
            valid_pool_launch = Pool(X_valid, y_valid, cat_features=cat_features)
            reg_launch_ = CatBoostRegressor(
                loss_function='MAE',
                has_time=config.has_time,
                eval_metric='MAE',
                iterations=3000,
                depth=config.depth,
                rsm=0.9,
                boost_from_average=True,
                reg_lambda=20.0,
                random_seed=0)
            reg_launch_.fit(train_pool_launch,
                           eval_set=valid_pool_launch,
                           use_best_model=True,
                           early_stopping_rounds=100,
                           verbose=config.verbose)

            reg_launch[fold] = reg_launch_
            y_pred_launch = reg_launch_.predict(valid_df)
            mae_launch = mean_absolute_error(y_pred_launch, valid_y.중식계)
            scores_launch.append(mae_launch)

            y_train, y_valid = train_y.석식계[train_idx], train_y.석식계[valid_idx]
            train_pool_dinner = Pool(X_train, y_train, cat_features=cat_features)
            valid_pool_dinner = Pool(X_valid, y_valid, cat_features=cat_features)
            reg_dinner_ = CatBoostRegressor(
                loss_function='MAE',
                has_time=config.has_time,
                eval_metric='MAE',
                iterations=3000,
                depth=config.depth,
                rsm=0.9,
                boost_from_average=True,
                reg_lambda=20.0,
                random_seed=0)
            reg_dinner_.fit(train_pool_dinner,
                           eval_set=valid_pool_dinner,
                           use_best_model=True,
                           early_stopping_rounds=100,
                           verbose=config.verbose)

            reg_dinner[fold] = reg_dinner_
            y_pred_dinner = reg_dinner_.predict(valid_df)
            mae_dinner = mean_absolute_error(y_pred_dinner, valid_y.석식계)
            scores_dinner.append(mae_dinner)
            print(f'================================================================================\n\n')

        print('-' * 50, 'Result', '-' * 50, '\n')
        print('Mean launch : {}'.format(np.mean(scores_launch, axis=0)))
        print('Mean dinner : {}'.format(np.mean(scores_dinner, axis=0)))
        print('Total : {}'.format((np.mean(scores_launch, axis=0) + np.mean(scores_dinner, axis=0)) / 2))
        print('_' * 106)

        return reg_launch, reg_dinner

    else :
        print('=' * 10, 'TRAIN LAUNCH MODEL', '=' * 10)
        train_pool_launch = Pool(train_df, train_y.중식계, cat_features=cat_features)
        valid_pool_launch = Pool(valid_df, valid_y.중식계, cat_features=cat_features)

        reg_launch = CatBoostRegressor(loss_function='MAE',
                                       has_time=True,
                                       eval_metric='MAE',
                                       iterations=3000,
                                       depth=config.depth,
                                       rsm=0.9,
                                       boost_from_average=True,
                                       reg_lambda=20.0,
                                       random_seed=0
                                       )

        reg_launch.fit(train_pool_launch,
                       eval_set=valid_pool_launch,
                       use_best_model=True,
                       early_stopping_rounds=100,
                       verbose=config.verbose)

        y_pred_launch = reg_launch.predict(valid_df)
        mae_launch = mean_absolute_error(y_pred_launch, valid_y.중식계)

        # ===========================================================================================
        print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)
        train_pool_launch = Pool(train_df, train_y.석식계, cat_features=cat_features)
        valid_pool_launch = Pool(valid_df, valid_y.석식계, cat_features=cat_features)

        reg_dinner = CatBoostRegressor(loss_function='MAE',
                                       has_time=True,
                                       eval_metric='MAE',
                                       iterations=3000,
                                       depth=config.depth,
                                       rsm=0.9,
                                       boost_from_average=True,
                                       reg_lambda=20.0,
                                       random_seed=0
                                       )

        reg_dinner.fit(train_pool_launch,
                       eval_set=valid_pool_launch,
                       use_best_model=True,
                       early_stopping_rounds=100,
                       verbose=config.verbose)

        y_pred_dinner = reg_dinner.predict(valid_df)
        mae_dinner = mean_absolute_error(y_pred_dinner, valid_y.석식계)

        print('=' * 10, 'RESULT', '=' * 10)
        print('MAE_LAUNCH :', mae_launch)
        print('MAE_DINNER :', mae_dinner)
        print('TOTAL SCORE :', (mae_launch + mae_dinner) / 2)
        print()

    return reg_launch, reg_dinner