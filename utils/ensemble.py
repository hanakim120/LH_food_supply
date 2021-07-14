import os

import pandas as pd


base = r'C:\Users\JCY\Dacon\pull\result'
fn_list = ['submission_lr.fs.seed0', 'submission_lr.fs.seed1', 'submission_lr.fs.seed2', 'submission_lr.fs.seed3', 'submission_lr.fs.seed4',
           'submission_lr.fs.seed5', 'submission_lr.fs.seed6']
# fn_list = ['submission_lr.seed0', 'submission_lr.seed1', 'submission_lr.seed2', 'submission_lr.seed3', 'submission_lr.seed4']

length = len(fn_list)
sample = pd.read_csv(r'C:\Users\JCY\Dacon\pull\data\sample_submission.csv', encoding='utf-8')

for fn in fn_list:
    path = os.path.join(base, fn + '.csv')
    file = pd.read_csv(path, encoding='utf-8')

    sample.중식계 += file.중식계
    sample.석식계 += file.석식계

sample.중식계 /= length
sample.석식계 /= length

sample.to_csv(r'C:\Users\JCY\Dacon\pull\result\result_ensemble.csv', index=False)
print(sample.head())