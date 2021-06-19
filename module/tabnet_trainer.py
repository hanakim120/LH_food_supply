import torch
import torch.optim as optim
import torch_optimizer as custom_optim
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


def train_tabnet(train_df, valid_df, train_y, valid_y, config):
    cat_idxs = [0, 2, 3, 4, 13, 15]
    le = LabelEncoder()
    for i in cat_idxs:
        train_df.iloc[:, i] = le.fit_transform(train_df.iloc[:, i])
        valid_df.iloc[:, i] = le.transform(valid_df.iloc[:, i])
    train_df = train_df.values
    valid_df = valid_df.values

    if config.lr_decay_start > 0:
        optimizer = optim.SGD
    else:
        if config.use_radam:
            optimizer = custom_optim.RAdam
        else:
            optimizer = optim.Adam

    print('OPTIMIZER : {}'.format(optimizer))

    print('=' * 10, 'TRAIN LUNCH MODEL', '=' * 10)
    reg_lunch = TabNetRegressor(
        n_d=config.n_d, # 8 - 64
        n_a=config.n_d,
        n_steps=config.n_steps, # 3 - 10
        gamma=config.gamma, # 1.3, relaxation parameter
        cat_idxs=cat_idxs,
        cat_dims=[5, 2, 2, 2, 12, 5],
        cat_emb_dim=[2, 1, 1, 1, 5, 2], # default:1
        n_independent=config.n_independent, # 1-5
        n_shared=config.n_shared,
        lambda_sparse=config.lambda_sparse,
        seed=0,
        optimizer_fn=optimizer,
        optimizer_params=dict(lr=config.optim_lr),
        clip_value=config.max_grad_norm,
        momentum=config.momentum,
        scheduler_fn=optim.lr_scheduler.MultiStepLR if config.lr_decay_start > 0 else None,
        scheduler_params=dict(
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                config.epochs,
                config.lr_step)],
            gamma=config.lr_gamma,
            last_epoch= -1,
        ) if config.lr_decay_start > 0 else None,
        device_name='cuda' if torch.cuda.is_available else 'cpu',
        verbose=config.verbose
    )

    reg_lunch.fit(
        train_df, train_y.중식계.values.reshape(-1, 1),
        eval_set=[(train_df, train_y.중식계.values.reshape(-1, 1)),
                  (valid_df, valid_y.중식계.values.reshape(-1, 1))],
        eval_name=['train', 'valid'],
        eval_metric=['mae'],
        patience=100,
        max_epochs=config.epochs,
        batch_size=config.batch_size,
        virtual_batch_size=config.batch_size // 5,
    )

    y_pred_lunch = reg_lunch.predict(valid_df)
    mae_lunch = mean_absolute_error(valid_y.중식계, y_pred_lunch)

    print('=' * 10, 'TRAIN DINNER MODEL', '=' * 10)
    reg_dinner = TabNetRegressor(
        n_d=config.n_d,  # 8 - 64
        n_a=config.n_d,
        n_steps=config.n_steps,  # 3 - 10
        gamma=config.gamma,  # 1.3, relaxation parameter
        cat_idxs=cat_idxs,
        cat_dims=[5, 2, 2, 2, 12, 5],
        cat_emb_dim=[2, 1, 1, 1, 5, 2],  # default:1
        n_independent=config.n_independent,  # 1-5
        n_shared=config.n_shared,
        lambda_sparse=config.lambda_sparse,
        seed=0,
        optimizer_fn=optimizer,
        optimizer_params=dict(lr=config.optim_lr),
        clip_value=config.max_grad_norm,
        momentum=config.momentum,
        scheduler_fn=optim.lr_scheduler.MultiStepLR if config.lr_decay_start > 0 else None,
        scheduler_params=dict(
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                config.epochs,
                config.lr_step)],
            gamma=config.lr_gamma,
            last_epoch=-1,
        ) if config.lr_decay_start > 0 else None,
        device_name='cuda' if torch.cuda.is_available else 'cpu',
        verbose=config.verbose
    )

    reg_dinner.fit(
        train_df, train_y.석식계.values.reshape(-1, 1),
        eval_set=[(train_df, train_y.석식계.values.reshape(-1, 1)),
                  (valid_df, valid_y.석식계.values.reshape(-1, 1))],
        eval_name=['train', 'valid'],
        eval_metric=['mae'],
        patience=100,
        max_epochs=config.epochs,
        batch_size=config.batch_size,
        virtual_batch_size=config.batch_size // 5,
    )

    y_pred_dinner = reg_dinner.predict(valid_df)
    mae_dinner = mean_absolute_error(valid_y.석식계, y_pred_dinner)

    print('=' * 10, 'RESULT', '=' * 10)
    print('MAE_LUNCH :', mae_lunch)
    print('MAE_DINNER :', mae_dinner)
    print('TOTAL SCORE :', (mae_lunch + mae_dinner) / 2)
    print()

    return reg_lunch, reg_dinner