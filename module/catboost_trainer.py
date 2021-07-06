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
    if not config.dummy_cat:
        cat_features = [0, 1, 9, 11, 16] # 요일, 공휴일전후, month, week, 체감온도, 폭염
    else:
        cat_features = []

    print('Columns: ', np.array(train_df.columns))
    print('Categorical Value: ', np.array(train_df.columns)[cat_features])

    if config.k > 0 :
        skf = KFold(n_splits=config.k, shuffle=True, random_state=config.seed)
        folds = []
        for train_idx, valid_idx in skf.split(train_df, train_y.중식계) :
            folds.append((train_idx, valid_idx))

        random.seed(42)
        reg_lunch = {}
        reg_dinner = {}
        scores_lunch = []
        scores_dinner = []
        for fold in range(config.k) :
            print(f'===================================={fold + 1}============================================')
            train_idx, valid_idx = folds[fold]
            X_train, X_valid, y_train, y_valid = train_df.drop(columns=['월저녁평균', '요일저녁평균']).iloc[train_idx], \
                                                 train_df.drop(columns=['월저녁평균', '요일저녁평균']).iloc[valid_idx], \
                                                 train_y.중식계.iloc[train_idx], train_y.중식계.iloc[valid_idx]

            train_pool_lunch = Pool(X_train, y_train, cat_features=cat_features)
            valid_pool_lunch = Pool(X_valid, y_valid, cat_features=cat_features)
            reg_lunch_ = CatBoostRegressor(
                loss_function='MAE',
                has_time=config.has_time,
                eval_metric='MAE',
                iterations=3000,
                depth=config.depth,
                rsm=config.rsm,
                boost_from_average=True,
                reg_lambda=20.0,
                random_seed=0)
            reg_lunch_.fit(train_pool_lunch,
                           eval_set=valid_pool_lunch,
                           use_best_model=True,
                           early_stopping_rounds=100,
                           verbose=config.verbose)

            reg_lunch[fold] = reg_lunch_
            y_pred_lunch = reg_lunch_.predict(valid_df)
            mae_lunch = mean_absolute_error(y_pred_lunch, valid_y.중식계)
            scores_lunch.append(mae_lunch)

            X_train, X_valid, y_train, y_valid = train_df.drop(columns=['월점심평균', '요일점심평균']).iloc[train_idx], \
                                                 train_df.drop(columns=['월점심평균', '요일점심평균']).iloc[valid_idx], \
                                                 train_y.석식계.iloc[train_idx], train_y.석식계.iloc[valid_idx]
            train_pool_dinner = Pool(X_train, y_train, cat_features=cat_features)
            valid_pool_dinner = Pool(X_valid, y_valid, cat_features=cat_features)
            reg_dinner_ = CatBoostRegressor(
                loss_function='MAE',
                has_time=config.has_time,
                eval_metric='MAE',
                iterations=3000,
                depth=config.depth,
                rsm=config.rsm,
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
        print('Mean lunch : {}'.format(np.mean(scores_lunch, axis=0)))
        print('Mean dinner : {}'.format(np.mean(scores_dinner, axis=0)))
        print('Total : {}'.format((np.mean(scores_lunch, axis=0) + np.mean(scores_dinner, axis=0)) / 2))
        print('_' * 106)

        return reg_lunch, reg_dinner

    else :
        print('=' * 10, 'TRAIN lunch MODEL', '=' * 10)
        train_pool_lunch = Pool(train_df.drop(columns=['월저녁평균','요일저녁평균']), train_y.중식계, cat_features=cat_features)
        valid_pool_lunch = Pool(valid_df.drop(columns=['월저녁평균','요일저녁평균']), valid_y.중식계, cat_features=cat_features)

        reg_lunch = CatBoostRegressor(loss_function='MAE',
                                       has_time=True,
                                       eval_metric='MAE',
                                       iterations=3000,
                                       depth=config.depth,
                                       rsm=config.rsm,
                                       boost_from_average=True,
                                       reg_lambda=20.0,
                                       random_seed=0
                                       )

        reg_lunch.fit(train_pool_lunch,
                       eval_set=valid_pool_lunch,
                       use_best_model=True,
                       early_stopping_rounds=100,
                       verbose=config.verbose)

        y_pred_lunch = reg_lunch.predict(valid_df)
        mae_lunch = mean_absolute_error(y_pred_lunch, valid_y.중식계)

        # ===========================================================================================
        print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)
        train_pool_lunch = Pool(train_df.drop(columns=['월점심평균','요일점심평균']), train_y.석식계, cat_features=cat_features)
        valid_pool_lunch = Pool(valid_df.drop(columns=['월점심평균','요일점심평균']), valid_y.석식계, cat_features=cat_features)

        reg_dinner = CatBoostRegressor(loss_function='MAE',
                                       has_time=True,
                                       eval_metric='MAE',
                                       iterations=3000,
                                       depth=config.depth,
                                       rsm=config.rsm,
                                       boost_from_average=True,
                                       reg_lambda=20.0,
                                       random_seed=0
                                       )

        reg_dinner.fit(train_pool_lunch,
                       eval_set=valid_pool_lunch,
                       use_best_model=True,
                       early_stopping_rounds=100,
                       verbose=config.verbose)

        y_pred_dinner = reg_dinner.predict(valid_df)
        mae_dinner = mean_absolute_error(y_pred_dinner, valid_y.석식계)

        print('=' * 10, 'RESULT', '=' * 10)
        print('MAE_lunch :', mae_lunch)
        print('MAE_DINNER :', mae_dinner)
        print('TOTAL SCORE :', (mae_lunch + mae_dinner) / 2)
        print()

    return reg_lunch, reg_dinner