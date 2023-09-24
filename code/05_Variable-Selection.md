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

# Analyze Correlations in the Data


## Intuition
* Want outcome that is strongly correlated with predictors
* Want predictors that are not correlated with each other
## Strategy
* Find the outcome(s) that is(are) most correlated with predictors
* Find the predictors that are correlated with the outcome, but not each other

<!-- #region pycharm={"name": "#%% md\n"} tags=[] -->
# Setup
<!-- #endregion -->

## Imports

```python
from plot_config import *

start_time = time()
```

## Import Data

```python
df = pd.read_csv(PLOT_SPECS['data_source']+'data_03.csv', low_memory=False)
for col in ['MONTH', 'ENROLL_DATE']:
    df[col] = df[col].astype('datetime64')
```

# Data Coverage

```python
df['SPA_PER_ID'].nunique()
```

```python
100 * df['SPA_PER_ID'].nunique() / 539339.0
```

```python
df[df['CMIS_MATCH']]['SPA_PER_ID'].nunique()
```

```python
100 * df[df['CMIS_MATCH']]['SPA_PER_ID'].nunique() / df['SPA_PER_ID'].nunique()
```

```python
df[~df['CMIS_MATCH']]['SPA_PER_ID'].nunique()
```

```python
df['SPA_ACCT_ID'].nunique()
```

```python
df['SPA_PREM_ID'].nunique()
```

# Spearman Correlation

```python
%%time
corr = df.drop(COLUMN_GROUPS['id']+COLUMN_GROUPS['problem'], axis=1).dropna().corr(method='spearman')
```

```python
sns.set(font_scale=0.5)
sns.heatmap(data=abs(corr.loc[COLUMN_GROUPS['avista']+COLUMN_GROUPS['city']+COLUMN_GROUPS['combined']+COLUMN_GROUPS['eng_pred']][COLUMN_GROUPS['outcome']+['WITHIN_12_MO']]), annot=True, fmt='.3f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.savefig(fname=PLOT_SPECS['image_dest']+'corr_zoom'+'.png', bbox_inches='tight')
plt.show();
```

Again, 'CMIS_MATCH' is the most associated with the predictor variables and 'WITHIN_6_MO' is 2nd.

```python
sns.set(font_scale=0.5)
sns.heatmap(data=abs(corr), annot=True, fmt='.1f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.savefig(fname=PLOT_SPECS['image_dest']+'corr'+'.png', bbox_inches='tight')
plt.show();
```

## Choose Outcome

```python
print(corr[~corr.index.isin(COLUMN_GROUPS['outcome'])][COLUMN_GROUPS['outcome']].sort_values(by='CMIS_MATCH', ascending=False).round(3).to_latex())
```

## Choose Predictors


### Groups of heatmaps

```python
billing_cols = [
    'RES_EL_CUR120_DAYS',
    'RES_EL_CUR22_DAYS',
    'RES_EL_CUR30_DAYS',
    'RES_EL_CUR60_DAYS',
    'RES_EL_CUR90_DAYS',
    'RES_EL_CUR_BAL_AMT',
    'RES_EL_OVER_120_DAYS',
    'RES_GAS_CUR120_DAYS',
    'RES_GAS_CUR22_DAYS',
    'RES_GAS_CUR30_DAYS',
    'RES_GAS_CUR60_DAYS',
    'RES_GAS_CUR90_DAYS',
    'RES_GAS_CUR_BAL_AMT',
    'RES_GAS_OVER_120_DAYS',
    'CITY_TOT_DUE',
    'CITY_30_DAYS_PAST_DUE_AMT',
    'CITY_60_DAYS_PAST_DUE_AMT',
    'CITY_90_DAYS_PAST_DUE_AMT',
    'AVISTA_CUR120_DAYS',
    'AVISTA_CUR22_DAYS',
    'AVISTA_CUR30_DAYS',
    'AVISTA_CUR60_DAYS',
    'AVISTA_CUR90_DAYS',
    'AVISTA_CUR_BAL_AMT',
    'AVISTA_OVER_120_DAYS',
    'TOTAL_CUR_BAL_AMT',
    'TOTAL_CUR30_DAYS',
    'TOTAL_CUR60_DAYS',
    'TOTAL_CUR90_DAYS',
]
corr_bill = df[billing_cols].corr(method='spearman')
```

```python
sns.set(font_scale=0.75)
sns.heatmap(data=abs(corr_bill), annot=True, fmt='.1f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.savefig(fname=PLOT_SPECS['image_dest']+'corr_bill'+'.png', bbox_inches='tight')
plt.show();
```

```python
other_cols = [
    'BREAK_ARRANGEMENT',
    'BREAK_PAY_PLAN',
    'CALL_OUT',
    'CALL_OUT_MANUAL',
    'DUE_DATE',
    'FINAL_NOTICE',
    'PAST_DUE',
    'SEVERANCE_ELECTRIC',
    'SEVERANCE_GAS',
    'COVID_REMINDER',
    'NUM_PREM_FOR_PER',
    'NUM_PER_FOR_PREM'
]
corr_other = df[other_cols].corr(method='spearman')
```

```python
sns.set(font_scale=0.75)
sns.heatmap(data=abs(corr_other), annot=True, fmt='.2f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.savefig(fname=PLOT_SPECS['image_dest']+'corr_other'+'.png', bbox_inches='tight')
plt.show();
```

### Running Lists

```python
running_list = abs(corr['CMIS_MATCH']).sort_values(ascending=False)
running_list = running_list[~running_list.index.isin(COLUMN_GROUPS['outcome'])]
running_list
```

Keep FINAL_NOTICE

```python
abs(corr['FINAL_NOTICE']).sort_values(ascending=False)
```

Do not keep any of:  
[PAST_DUE, CALL_OUT, DUE_DATE, SEVERANCE_ELECTRIC]

```python
running_list = running_list[~running_list.index.isin(['PAST_DUE', 'CALL_OUT', 'DUE_DATE', 'SEVERANCE_ELECTRIC'])]
running_list
```

```python
temp = corr.loc[COLUMN_GROUPS['avista']+COLUMN_GROUPS['city']][COLUMN_GROUPS['avista']+COLUMN_GROUPS['city']]
temp[temp < 1.0].describe().transpose()
```

Because the non-aggregaed amounts owed contain more information than the aggregated, and because they are relatively uncorrelated with each other, the non-aggregated amounts owed will be retained and the aggregated amounts owed will be removed.

```python
running_list = running_list[~running_list.index.isin(COLUMN_GROUPS['avista_agg']+COLUMN_GROUPS['combined'])]
running_list
```

```python
abs(corr['NUM_PER_FOR_PREM']).loc[running_list.index].sort_values(ascending=False)
```

```python
abs(corr['BREAK_PAY_PLAN']).loc[running_list.index].sort_values(ascending=False)
```

Remove 'BREAK_ARRANGEMENT'

```python
running_list = running_list[~(running_list.index == 'BREAK_ARRANGEMENT')]
running_list
```

```python
abs(corr['CALL_OUT_MANUAL']).loc[running_list.index].sort_values(ascending=False)
```

```python
abs(corr['SEVERANCE_GAS']).loc[running_list.index].sort_values(ascending=False)
```

Remove 'COVID_REMINDER', only applicable to 2020

```python
running_list = running_list[~(running_list.index == 'COVID_REMINDER')]
running_list
```

```python
running_list.index.tolist()
```

# Save

```python
df[running_list.index.tolist()+['CMIS_MATCH', 'WITHIN_12_MO']].to_csv(PLOT_SPECS['data_source']+'data_05.csv', index=False)
```

# Time

```python
print(time() - start_time)
```
