---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

Resources:
* https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html#statsmodels.discrete.discrete_model.Logit
* https://www.andrewvillazon.com/logistic-regression-python-statsmodels/#fitting-a-logistic-regression

```python
from plot_config import *
import model_utils
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from copy import copy

RANDOM_SEED = 23
rng = np.random.default_rng(seed=RANDOM_SEED)
start_time = time()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# Load Data
<!-- #endregion -->

```python
df = pd.read_csv(PLOT_SPECS['data_source']+'data_03.csv', low_memory=False)

for col in ['MONTH', 'ENROLL_DATE']:
    df[col] = df[col].astype('datetime64')
```

```python
# df.groupby('CALL_OUT')['WITHIN_12_MO'].sum()
```

```python
# df.groupby('FINAL_NOTICE')['WITHIN_12_MO'].sum().loc[[3, 4]]
```

```python
# col = 'WITHIN_12_MO'
# temp = pd.DataFrame(index=df['PAST_DUE'].unique(), columns=['n_pos', 'n_tot', 'recall', 'precision', 'prop_pop'])
# p_tot = df[col].sum()
# tot = len(df)
# for past_due in temp.index:
#     temp.loc[past_due, 'n_pos'] = df[df['PAST_DUE'] >= past_due][col].sum()
#     temp.loc[past_due, 'n_tot'] = len(df[df['PAST_DUE'] >= past_due])
#     temp.loc[past_due, 'recall'] = temp.loc[past_due, 'n_pos'] / p_tot
#     temp.loc[past_due, 'precision'] = temp.loc[past_due, 'n_pos'] / temp.loc[past_due, 'n_tot']
#     temp.loc[past_due, 'prop_pop'] = temp.loc[past_due, 'n_tot'] / tot

# temp = temp.sort_index()
# temp.index = [f'>={x}' for x in temp.index]
# temp.index.name = 'PAST_DUE'

# temp.to_csv(PLOT_SPECS['data_source']+'temp.csv', index=True)
# temp
```

```python
# df.pivot_table(index=['PAST_DUE'], columns=['CALL_OUT'], values=['WITHIN_12_MO'], aggfunc='sum')
```

```python
'''
predictors = [
    'FINAL_NOTICE',
    'NUM_PER_FOR_PREM',
    'RES_EL_CUR60_DAYS',
    'CITY_60_DAYS_PAST_DUE_AMT',
    'RES_GAS_CUR60_DAYS',
    'RES_EL_CUR90_DAYS',
    'CITY_90_DAYS_PAST_DUE_AMT',
    'CITY_30_DAYS_PAST_DUE_AMT',
    'RES_EL_CUR_BAL_AMT',
    'RES_GAS_CUR90_DAYS',
    'CITY_TOT_DUE',
    'BREAK_PAY_PLAN',
    'RES_EL_CUR120_DAYS',
    'RES_EL_CUR30_DAYS',
    'RES_GAS_CUR120_DAYS',
    'RES_GAS_CUR30_DAYS',
    'RES_EL_OVER_120_DAYS',
    'RES_GAS_CUR_BAL_AMT',
    'RES_EL_CUR22_DAYS',
    'RES_GAS_OVER_120_DAYS',
    'NUM_PREM_FOR_PER',
    'RES_GAS_CUR22_DAYS',
    'CALL_OUT_MANUAL',
    'SEVERANCE_GAS'
]
'''
predictors = [
    'PAST_DUE',
    'TOTAL_CUR_BAL_AMT',
    'NUM_PREM_FOR_PER',
    'NUM_PER_FOR_PREM',
    'BREAK_ARRANGEMENT'
]

```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# Data Stats
<!-- #endregion -->

```python
# print(df.drop(COLUMN_GROUPS['outcome']+COLUMN_GROUPS['id']+COLUMN_GROUPS['problem'], axis=1).describe().transpose()[['min', 'max', 'mean', 'std']].sort_values(by='mean').to_latex(float_format='%.2e'))
```

```python
print(df['MONTH'].min(), df['MONTH'].max())
```

```python
print(f"{df[df['CMIS_MATCH']]['SPA_PER_ID'].nunique()} = {100 * df[df['CMIS_MATCH']]['SPA_PER_ID'].nunique() / df['SPA_PER_ID'].nunique():0.3f}%")
```

```python
print(f"{df['WITHIN_12_MO'].sum()} = {100 * df['WITHIN_12_MO'].sum() / len(df):0.3f}%")
```

```python
df[~df['WITHIN_12_MO']]['SPA_PER_ID'].nunique()
```

```python
df['SPA_PER_ID'].nunique()
```

```python
100 * df['SPA_PER_ID'].nunique() / 539339
```

```python
df['SPA_PREM_ID'].nunique()
```

```python
df['SPA_ACCT_ID'].nunique()
```

# Fit Models

```python
%%time

n_folds = 10
n_repeats = 10
thresh = 0.8

X, y, repeats, check = model_utils.get_folds(
    n_folds=n_folds, 
    n_repeats=n_repeats, 
    data=df, 
    x_cols=predictors,
    y_col='WITHIN_12_MO', 
    random_seed=RANDOM_SEED
)
```

```python
%%time
test_ratio = 0.25
tracker = {
    'models': [],
    'scalers': [],
    'samplers': [],
    'metrics': [],
    'test_index': []
}
failed_to_converge = []
predictions = {}

step = 0
for i in range(n_repeats):
    for j in range(n_folds):
        fold = repeats[i][j]
        rng.shuffle(fold)
        test_size = int(test_ratio * len(fold))
        train_index = fold[:-test_size]
        test_index = fold[-test_size:]

        X_train, X_test = copy(X[train_index]), copy(X[test_index])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train, y_test = y[train_index], y[test_index]
        sampler = SMOTE()
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        try:
            model = Logit(exog=X_train, endog=y_train).fit(maxiter=1000, disp=False)
            y_pred = model.predict(X_test)

            tracker['models'].append(model)
            tracker['scalers'].append(scaler)
            tracker['samplers'].append(sampler)
            tracker['metrics'].append(model_utils.get_metrics(y_true=y_test, y_pred=y_pred, thresh=thresh))
            tracker['test_index'].append(test_index)

            predictions['y_pred_'+str(step)] = y_pred
            predictions['y_true_'+str(step)] = y_test

            print('Completed step', step, '/', n_repeats*n_folds)
        except:
            failed_to_converge.append(test_index)
            print('Failed to converge on step', step, '/', n_repeats*n_folds)
        step += 1
```

```python
# %%time
# test_ratio = 0.25
# tracker = {
#     'models': [],
#     'scalers': [],
#     'samplers': [],
#     'metrics': [],
#     'test_index': []
# }
# failed_to_converge = []
# predictions = {}

# step = 0
# for i in range(n_repeats):
#     for j in range(n_folds):
#         fold = repeats[i][j]
#         rng.shuffle(fold)
#         test_size = int(test_ratio * len(fold))
#         train_index = fold[:-test_size]
#         test_index = fold[-test_size:]

#         X_train, X_test = copy(X[train_index]), copy(X[test_index])
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
#         y_train, y_test = y[train_index], y[test_index]
#         sampler = SMOTE()
#         X_train, y_train = sampler.fit_resample(X_train, y_train)
#         try:
#             model = Logit(exog=X_train, endog=y_train).fit(maxiter=1000, disp=False)
#             y_pred = model.predict(X_test)

#             tracker['models'].append(model)
#             tracker['scalers'].append(scaler)
#             tracker['samplers'].append(sampler)
#             tracker['metrics'].append(model_utils.get_metrics(y_true=y_test, y_pred=y_pred))
#             tracker['test_index'].append(test_index)

#             predictions['y_pred_'+str(step)] = y_pred
#             predictions['y_true_'+str(step)] = y_test

#             print('Completed step', step, '/', n_repeats*n_folds)
#         except:
#             failed_to_converge.append(test_index)
#             print('Failed to converge on step', step, '/', n_repeats*n_folds)
#         step += 1
```

```python
# Transform the predictions to a DataFrame
max_len = 0
for k in predictions:
    max_len = max(max_len, len(predictions[k]))
for k in predictions:
    gap = max_len - len(predictions[k])
    predictions[k] = np.concatenate([predictions[k], [np.nan]*gap])

predictions = pd.DataFrame(data=predictions)
```

# Results

```python
# Get summary metrics
summary = {k:[] for k in tracker['metrics'][0].keys() if k != 'confusion_matrix'}

for i in range(len(tracker['metrics'])):
    for metric in summary.keys():
        summary[metric].append(tracker['metrics'][i][metric])

for metric in summary.keys():
    summary[metric] = (np.mean(summary[metric]), np.std(summary[metric]))

summary = pd.DataFrame(data=summary).transpose()
summary = summary.rename(columns={0: 'Mean', 1: 'Std'})
summary.index = [idx.upper() for idx in summary.index]
# print(summary.to_latex(float_format='%.3f'))
print(summary.to_string())
```

```python
# Get confusion matrix
mat = pd.DataFrame(data=0.0, index=['f_act', 't_act'], columns=['f_pred', 't_pred'], dtype=float)

for i in range(len(tracker['metrics'])):
    mat = mat + tracker['metrics'][i]['confusion_matrix']

mat = mat / (len(tracker['metrics']))
mat = mat.reindex(index=['t_act', 'f_act'], columns=['t_pred', 'f_pred'])
total = mat.sum().sum()
mat = 100 * mat / total
# print(mat.to_latex(float_format='%.3f'))
print(mat.to_string())
```

```python
# # Get params
# params = pd.DataFrame(index=predictors,columns=['mean', 'std'])
# arr = np.zeros(shape=(len(tracker['models']), len(params.index)))
# for i in range(len(tracker['models'])):
#     arr[i, :] = tracker['models'][i].params
# for i in range(len(params.index)):
#     params.iloc[i, 0] = np.mean(arr[:, i])
#     params.iloc[i, 1] = np.std(arr[:, i])
# params = params.sort_values(by='mean', ascending=False).astype(float)
# print(params.to_latex(float_format='%.4f'))
```

# Save Predictions

```python
# predictions.to_csv(PLOT_SPECS['data_source']+'predictions_log.csv', index=False)
```

# Check Overall Model

```python
# scaler = StandardScaler()
# X = scaler.fit_transform(df[predictors])
# y = df['WITHIN_12_MO'].copy()
# sampler = SMOTE()
# X, y = sampler.fit_resample(X, y)

# model = Logit(exog=X, endog=y).fit(maxiter=1000, disp=True)
```

```python
# params_all = model.params
# params_all.index = predictors
# params_all = params_all.to_frame()
# params_all[['lower', 'upper']] = model.conf_int(alpha=0.05).values
# params_all['p-value'] = model.pvalues.values
# params_all = params_all.rename(columns={0: 'coef'})
# params_all = params_all.sort_values(by='coef', ascending=False)
# print(params_all.to_latex(float_format='%.4f'))
```

```python
# params_all['p-value']
```

# Time

```python
print(time() - start_time)
```
