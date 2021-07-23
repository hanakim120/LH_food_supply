# 🎊Quick start🎉
Got 14th(top 3%) on public score.
```buildoutcfg
python train.py --model lr --holiday_length --weather --outlier
```

## Training
```buildoutcfg
python train.py --model catboost --depth 3 --lr 1e-1
```

```buildoutcfg
========== PCA RESULT ==========
breakfast explained variance [0.25148388 0.37367311 0.46665427]
lunch explained variance [0.2620023  0.36952975 0.46542106]
dinner explained variance [0.26094099 0.39950021 0.48948373]

========== MESSAGE: DATA LOADED SUCCESSFULLY ==========
|TRAIN| : (1000, 29) |VALID| : (205, 29) |TEST| : (50, 29)
Missing values: 0

Columns:  ['요일' '공휴일전후' '공휴일길이' '출근' '휴가비율' '출장비율' '야근비율' '재택비율' 'year' 'month'
 'day' 'week' '강수량' '불쾌지수' '체감온도' '요일점심평균' '요일저녁평균' '월점심평균' '월저녁평균'
 '전일대비증감' 0 1 2 3 4 5 6 7 8]
========== TRAIN LUNCH MODEL ==========
(Lunch) train score :  68.1250047243973
        valid score:  76.92316808031497

========== TRAIN DINNER MODEL ==========
(Dinner) train score :  63.01406437427006
         valid score:  57.88981133475012

========== RESULT ==========
TOTAL SCORE : 67.40648970753254

========== SAVE COMPLETED ==========
           일자          중식계         석식계
0  2021-01-27  1026.794636  342.914369
1  2021-01-28   980.453254  464.353275
2  2021-01-29   673.898736  230.204398
3  2021-02-01  1249.020407  520.190438
4  2021-02-02  1073.293711  535.185174

```

# 📐Parameter tuning📚
## Common
1. **model**: model selection. result file will be saved as "./data/submission_[model].csv"
              (catboost / lgbm / lr / rf / reg / tabnet)
2. **text**: Method to preprocess text data (default: menu) 
             (embedding / menu / autoencode)
3. **menu_fn**: menu embedding file name (csv format; default: embedding.oov)
4. **train_size**: The number of datas which will be used for training
5. **dim**: embedding dimension
6. **pca_dim**: conduct PCA on the whole data when this parameter is larger than 0
7. **dummy_cat**: transform categorical feateatures into dummy variables   
8. **seed**: random seed
9. **k**: using k-fold when k > 0
10. **verbose**: verbosity during training
11. **epochs**: epochs

## Features
1. **oov_cnt**: add the number of OOV(out of vocabulary) as a feature
2. **weather**: add weather related features
3. **holiday_length**: add holiday length before a day   
4. **dust**: add dust feature  
5. **feature_selection**: use only statistically valid features (cat / multicol / None)


## fasttext
1. **fasttext_model_fn**: fasttext model file path (train new model if path invalid)
2. **pretrained**: Use pretrained korean language model (supported by fasttext)
3. **use_tok**: use mecab tokenizing when embedding text data (Avalibable only if mecab installed on your environment)
4. **min_count**
5. **window_size**
6. **min_ngram**
7. **max_ngram**
8. **fasttext_epoch**: number of training epochs
9. **dummy_corpus**: using dummy data

## CatBoost
1. **has_time**: whether randomly shuffle datas or not
2. **depth**: growth stop criterion
3. **rsm**: random subsampling ratio

## Tabnet
1. **n_d**: Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting. Values typically range from 8 to 64. (n_d == n_a)
2. **n_steps**: Number of steps in the architecture (usually between 3 and 10)
3. **gamma**: This is the coefficient for feature reusage in the masks. A value close to 1 will make mask selection least correlated between layers. Values range from 1.0 to 2.0.
4. **lambda_sparse**: This is the extra sparsity loss coefficient as proposed in the original paper. The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help.
5. **n_independent**: Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.
6. **n_shared**: Number of shared Gated Linear Units at each step Usual values range from 1 to 5
7. **use_radam**: use radam as optimizer for training tabnet, oterwise use adam instead.
8. **epochs**: epochs
9. **batch_size**: batch size
10. **lr_decay_start**: when to start learning rate decay
11. **lr_step**: apply learning rate decay with every N step
12. **lr_gamma**: learning rate reduction ratio
13. **optim_lr**: initialize learning rate for optimizer

## LGBM
1. **num_leaves**: number of leaves limitation for growth stopping
2. **min_data_in_leaf**: minimum number of data in leaf when the tree growing
3. **min_sum_hessian_in_leaf**: minimum sum of hessian(gradient of gradient)
4. **max_bin**: maximum number of bin for features
5. **min_data_in_bin**: minimum data in bin for features
6. **max_cat_threshold**: maximum threshold for categorical feature treatment
7. **lgbm_epoch**: number of iteration(=number of estimators)
8. **lr**: earning rate
9. **depth**: growth stop criterion

## Random Forest
1. **n_est**: The number of estimators
2. **min_samp**: minimum number of sample to split node

## Regularized Regression
1. **alpha**

## Experimental
1. **sum_reduction**: sum all embedding values and divide into embedding dimension
2. **drop_text**: drop text columns (set methods to "embedding")