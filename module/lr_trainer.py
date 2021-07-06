import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


def train_lr(train_df,
                    valid_df,
                    train_y,
                    valid_y,
                    config):

    print('Columns: ', np.array(train_df.columns))

    print('=' * 10, 'TRAIN LUNCH MODEL', '=' * 10)

    reg_lunch = LinearRegression(normalize=True)

    reg_lunch.fit(train_df.drop(columns=['월저녁평균','요일저녁평균']), train_y.중식계)

    y_pred_lunch = reg_lunch.predict(valid_df.drop(columns=['월저녁평균','요일저녁평균']))
    mae_lunch = mean_absolute_error(y_pred_lunch, valid_y.중식계)

    print('(Lunch) train score : ', mean_absolute_error(reg_lunch.predict(train_df.drop(columns=['월저녁평균', '요일저녁평균'])), train_y.중식계))
    print('        valid score: ', mae_lunch)
    print()


    # ===========================================================================================
    print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)

    reg_dinner = LinearRegression(normalize=True)
    reg_dinner.fit(train_df.drop(columns=['월점심평균','요일점심평균']), train_y.석식계)

    y_pred_dinner = reg_dinner.predict(valid_df.drop(columns=['월점심평균','요일점심평균']))
    mae_dinner = mean_absolute_error(y_pred_dinner, valid_y.석식계)
    print('(Dinner) train score : ', mean_absolute_error(reg_dinner.predict(train_df.drop(columns=['월점심평균', '요일점심평균'])), train_y.석식계))
    print('         valid score: ', mae_dinner)
    print()

    total_score = (mae_lunch + mae_dinner) / 2

    print('=' * 10, 'RESULT', '=' * 10)
    print('TOTAL SCORE :', total_score)
    print()

    return reg_lunch, reg_dinner