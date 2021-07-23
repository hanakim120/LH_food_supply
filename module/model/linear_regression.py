import random

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def train_lr(
        train_df,
        valid_df,
        train_y,
        valid_y,
        config,
        lunch_drop_col,
        dinner_drop_col
         ):

    print('Lunch Columns: ', np.array(train_df.drop(columns=lunch_drop_col).columns))
    print('Dinner Columns: ', np.array(train_df.drop(columns=dinner_drop_col).columns))

    if config.k > 0 :
        train_df = pd.concat([train_df, valid_df], axis=0)
        train_y = pd.concat([train_y, valid_y], axis=0)

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
            X_train, X_valid, y_train, y_valid = train_df.drop(columns=lunch_drop_col).iloc[train_idx], \
                                                 train_df.drop(columns=lunch_drop_col).iloc[valid_idx], \
                                                 train_y.중식계.iloc[train_idx], train_y.중식계.iloc[valid_idx]

            reg_lunch_ = LinearRegression()
            reg_lunch_.fit(X_train, y_train)

            reg_lunch[fold] = reg_lunch_
            y_pred_lunch = reg_lunch_.predict(X_valid)
            mae_lunch = mean_absolute_error(y_pred_lunch, y_valid)
            scores_lunch.append(mae_lunch)

            X_train, X_valid, y_train, y_valid = train_df.drop(columns=dinner_drop_col).iloc[train_idx], \
                                                 train_df.drop(columns=dinner_drop_col).iloc[valid_idx], \
                                                 train_y.석식계.iloc[train_idx], train_y.석식계.iloc[valid_idx]
            reg_dinner_ = LinearRegression()
            reg_dinner_.fit(X_train, y_train)

            reg_dinner[fold] = reg_dinner_
            y_pred_dinner = reg_dinner_.predict(X_valid)
            mae_dinner = mean_absolute_error(y_pred_dinner, y_valid)
            scores_dinner.append(mae_dinner)
            print(f'================================================================================\n\n')

        print('-' * 50, 'Result', '-' * 50, '\n')
        print('Mean lunch : {}'.format(np.mean(scores_lunch, axis=0)))
        print('Mean dinner : {}'.format(np.mean(scores_dinner, axis=0)))
        print('Total : {}'.format((np.mean(scores_lunch, axis=0) + np.mean(scores_dinner, axis=0)) / 2))
        print('_' * 106)

        return reg_lunch, reg_dinner

    else :
        print('=' * 10, 'TRAIN LUNCH MODEL', '=' * 10)

        reg_lunch = LinearRegression()

        reg_lunch.fit(train_df.drop(columns=lunch_drop_col), train_y.중식계)

        y_pred_lunch = reg_lunch.predict(valid_df.drop(columns=lunch_drop_col))
        mae_lunch = mean_absolute_error(y_pred_lunch, valid_y.중식계)

        print('(Lunch) train score : ', mean_absolute_error(reg_lunch.predict(train_df.drop(columns=lunch_drop_col)), train_y.중식계))
        print('        valid score: ', mae_lunch)
        print()


        # ===========================================================================================
        print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)

        reg_dinner = LinearRegression()
        reg_dinner.fit(train_df.drop(columns=dinner_drop_col), train_y.석식계)

        y_pred_dinner = reg_dinner.predict(valid_df.drop(columns=dinner_drop_col))
        mae_dinner = mean_absolute_error(y_pred_dinner, valid_y.석식계)
        print('(Dinner) train score : ', mean_absolute_error(reg_dinner.predict(train_df.drop(columns=dinner_drop_col)), train_y.석식계))
        print('         valid score: ', mae_dinner)
        print()

        total_score = (mae_lunch + mae_dinner) / 2

        print('=' * 10, 'RESULT', '=' * 10)
        print('TOTAL SCORE :', total_score)
        print()

    return reg_lunch, reg_dinner