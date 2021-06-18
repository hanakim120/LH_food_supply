# 🎊Quick start🎉
## Training
```buildoutcfg
>>python train.py --model catboost --fasttext_epoch 2000 --sep_date --dummy_corpus
```
or
```buildoutcfg
>>python train.py --model lgbm --depth 3 --sep_date --dummy_corpus --k 5 --lr 0.04

```buildoutcfg
Categorical Features : [0, 6, 7]
========== TRAIN LAUNCH MODEL ==========
0:      learn: 167.0310572      test: 153.5968870       best: 153.5968870 (0)   total: 143ms    remaining: 7m 9s
100:    learn: 80.7203422       test: 76.3503910        best: 76.3503910 (100)  total: 830ms    remaining: 23.8s
200:    learn: 70.6280619       test: 69.6358031        best: 69.6358031 (200)  total: 1.51s    remaining: 21s
300:    learn: 65.1974062       test: 67.5897186        best: 67.5897186 (300)  total: 2.3s     remaining: 20.6s
400:    learn: 61.7769994       test: 66.9983577        best: 66.9983577 (400)  total: 3.06s    remaining: 19.8s
500:    learn: 59.2809029       test: 66.7950628        best: 66.7295371 (474)  total: 3.79s    remaining: 18.9s
Stopped by overfitting detector  (100 iterations wait)

bestTest = 66.72953711
bestIteration = 474

Shrink model to first 475 iterations.
========== TRAIN DINNER MODEL ==========
0:      learn: 99.1778206       test: 90.2731526        best: 90.2731526 (0)    total: 7.44ms   remaining: 22.3s
100:    learn: 65.6623150       test: 57.0599211        best: 57.0599211 (100)  total: 822ms    remaining: 23.6s
200:    learn: 53.6771937       test: 47.3983845        best: 47.3983845 (200)  total: 1.6s     remaining: 22.3s
300:    learn: 48.6255771       test: 44.8366343        best: 44.8366343 (300)  total: 2.29s    remaining: 20.6s
400:    learn: 45.9185217       test: 44.4320334        best: 44.4271799 (392)  total: 3.04s    remaining: 19.7s
500:    learn: 43.8352190       test: 44.1411717        best: 44.1370020 (499)  total: 3.75s    remaining: 18.7s
600:    learn: 42.0927856       test: 44.0551806        best: 43.9661026 (572)  total: 4.51s    remaining: 18s
Stopped by overfitting detector  (100 iterations wait)

bestTest = 43.96610264
bestIteration = 572

Shrink model to first 573 iterations.
========== RESULT ==========
MAE_LAUNCH : 66.7295381089314
MAE_DINNER : 43.966103641137856
TOTAL SCORE : 55.34782087503463

========== SAVE COMPLETED ==========
           일자       중식계      석식계
45  2021-04-05  1209.855948  554.705740
46  2021-04-06  1033.257364  543.827706
47  2021-04-07   941.435875  372.935169
48  2021-04-08   862.466410  420.880682
49  2021-04-09   632.735706  326.675741

```

## Making dummy data
```buildoutcfg
>>python ./utils/dummy.py --load_fn ./data/corpus.train.txt --save_fn ./data/corpus.train.dummy.txt --iter 5 --verbose 300
```
```buildoutcfg
|ITERATION 1|
Train completed on 0 datas
Train completed on 300 datas
Train completed on 600 datas
Train completed on 900 datas
Train completed on 1200 datas
Train completed on 1500 datas
Train completed on 1800 datas
Train completed on 2100 datas
Train completed on 2400 datas
Train completed on 2700 datas
...
|ITERATION 5|
Train completed on 0 datas
Train completed on 300 datas
Train completed on 600 datas
Train completed on 900 datas
Train completed on 1200 datas
Train completed on 1500 datas
Train completed on 1800 datas
Train completed on 2100 datas
Train completed on 2400 datas
Train completed on 2700 datas
COMPLETED: 14721 data is augmented

```

# 📐Parameter tuning📚
## Common
1. **model**: model selection (catboost, ...)
2. **text**: Method to preprocess text data (embedding / tokenize / raw)
3. **dummy_corpus**: making dummy data using random permutation
4. **seed**: random seed

## fasttext
1. **fasttext_model_fn**: fasttext model file path (train new model if path invalid)
2. **pretrained**: Use pretrained korean language model (supported by fasttext)
3. **use_tok**: use mecab tokenizing when embedding text data (Avalibable only if mecab installed on your environment)
4. **dim**: embedding dimension (only woking when using embedding method)
5. **min_count**
6. **window_size**
7. **min_ngram**
8. **max_ngram**
9. **fasttext_epoch**: number of training epochs

## CatBoost
1. **has_time**: whether randomly shuffle datas or not
2. **depth**: growth stop criterion
3. **verbose**: verbosity
4. **k**: using k-fold when k > 0

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
10. **k**: using k-fold when k > 0
11. **verbose**: verbosity
12. **seed**: random seed

## Experimental
1. **sum_reduction**: sum all embedding values and divide into embedding dimension
2. **drop_text**: drop text columns (set methods to "embedding")