import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso, Ridge

def train_reg_model(train_df,
                    valid_df,
                    train_y,
                    valid_y,
                    config):

    print('Columns: ', np.array(train_df.columns))

    params = {'alpha': config.alpha,
              'max_iter': config.epochs,
              'normalize': True,
              'random_state': config.seed}

    scores = []
    reg_lunchs = []
    reg_dinners = []
    model_list = [('l1', Lasso),
                  ('l2', Ridge)]

    for name, model in model_list:
        print('=' * 10, 'TRAIN LUNCH {}'.format(name), '=' * 10)

        reg_lunch = model(**params)

        reg_lunch.fit(train_df.drop(columns=['월저녁평균','요일저녁평균']), train_y.중식계)

        y_pred_lunch = reg_lunch.predict(valid_df.drop(columns=['월저녁평균','요일저녁평균']))
        mae_lunch = mean_absolute_error(y_pred_lunch, valid_y.중식계)

        print('(Lunch) train score : ', mean_absolute_error(reg_lunch.predict(train_df.drop(columns=['월저녁평균', '요일저녁평균'])), train_y.중식계))
        print('        valid score: ', mae_lunch)
        print()


        # ===========================================================================================
        print('=' * 10, 'TRAIN DINNER {}'.format(name), '=' * 10)

        reg_dinner = model(**params)
        reg_dinner.fit(train_df.drop(columns=['월점심평균','요일점심평균']), train_y.석식계)

        y_pred_dinner = reg_dinner.predict(valid_df.drop(columns=['월점심평균','요일점심평균']))
        mae_dinner = mean_absolute_error(y_pred_dinner, valid_y.석식계)
        print('(Dinner) train score : ', mean_absolute_error(reg_dinner.predict(train_df.drop(columns=['월점심평균', '요일점심평균'])), train_y.석식계))
        print('         valid score: ', mae_dinner)
        print()

        total_score = (mae_lunch + mae_dinner) / 2

        print('=' * 10, 'RESULT', '=' * 10)
        print('TOTAL SCORE :', total_score)
        scores.append(total_score)
        reg_lunchs.append(reg_lunch)
        reg_dinners.append(reg_dinner)
        print()

    if scores[0] < scores[1]:
        print('{} model get lower mae, score : {}'.format(model_list[0][0], scores[0]))
        reg_lunch, reg_dinner = reg_lunchs[0], reg_dinners[0]
    else:
        print('{} model get lower mae, score : {}'.format(model_list[1][0], scores[1]))
        reg_lunch, reg_dinner = reg_lunchs[1], reg_dinners[1]

    return reg_lunch, reg_dinner