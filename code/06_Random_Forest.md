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
from model_utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 4223
rng = np.random.default_rng(seed=RANDOM_SEED)
start_time = time()
```

```python
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

df = pd.read_csv(PLOT_SPECS['data_source']+'data_03.csv', low_memory=False)
for col in ['MONTH', 'ENROLL_DATE']:
    df[col] = df[col].astype('datetime64')
```

```python
#num_prems = df.groupby(['SPA_PER_ID', 'MONTH'])['SPA_PREM_ID'].nunique().to_frame()
#num_prems = num_prems.rename(columns={'SPA_PREM_ID': 'NUM_PREM_FOR_MO'})
#num_prems = num_prems.join(df.set_index(['SPA_PER_ID', 'MONTH'])[['CMIS_MATCH', 'WITHIN_6_MO']], how='left')
# num_prems.head()
```

```python
# num_prems[num_prems['WITHIN_6_MO']]['NUM_PREM_FOR_MO'].value_counts()
```

```python
# num_prems[~num_prems['WITHIN_6_MO']]['NUM_PREM_FOR_MO'].value_counts()
```

```python
# num_prems[['WITHIN_6_MO', 'CMIS_MATCH', 'NUM_PREM_FOR_MO']].corr(method='spearman')
```

```python
#df = df.set_index(['SPA_PER_ID', 'MONTH'])
#df = df.join(num_prems['NUM_PREM_FOR_MO'], how='left')
#df = df.reset_index()
```

```python
%%time
# The default values in scikit-learn's RepeatedStratifiedKFold
n_folds = 5
n_repeats = 10
X, y, repeats, check = get_folds(
    n_folds=n_folds, 
    n_repeats=n_repeats, 
    data=df, 
    x_cols=predictors, #+['NUM_PREM_FOR_MO'], 
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
        try:
            model = RandomForestClassifier(random_state=RANDOM_SEED).fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 0]

            tracker['models'].append(model)
            tracker['scalers'].append(scaler)
            tracker['metrics'].append(get_metrics(y_true=y_test, y_pred=y_pred))
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
summary = summary.rename(columns={0: 'mean', 1: 'std'})
print(summary.to_latex(float_format='%.3f'))
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
print(mat.to_latex(float_format='%.2f'))
```

```python
# Get params
params = pd.DataFrame(
    index=predictors, #+['NUM_PREM_FOR_MO'], 
    columns=['mean', 'std']
)
arr = np.zeros(shape=(len(tracker['models']), len(params.index)))
for i in range(len(tracker['models'])):
    arr[i, :] = tracker['models'][i].params
for i in range(len(params.index)):
    params.iloc[i, 0] = np.mean(arr[:, i])
    params.iloc[i, 1] = np.std(arr[:, i])
params = params.sort_values(by='mean', ascending=False).astype(float)
print(params.to_latex(float_format='%.4f'))
```

# Save Predictions

```python
predictions.to_csv(PLOT_SPECS['data_source']+'predictions_rfc.csv', index=False)
```

# Time

```python
print(time() - start_time)
```
