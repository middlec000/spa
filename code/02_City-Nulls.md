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

# How to deal with null values in the city billing columns?


There are null values in some of the city billing data. Either
* Only use Avista billing data
* Drop the null rows and use both  

Use Spearman's Correlation to evaluate which division of data contains stronger correlations to the outcome variables.  

Spearman's Correlation is non-directional, meaning $\rho(row, col)=\rho(col, row)$

<!-- #region pycharm={"name": "#%% md\n"} tags=[] -->
# Setup
<!-- #endregion -->

## Imports

```python
from plot_config import *

start = time()
```

## Import Data

```python
df = pd.read_csv(PLOT_SPECS['data_source']+'data_01.csv', low_memory=False)
for col in ['MONTH', 'ENROLL_DATE']:
    df[col] = df[col].astype('datetime64')
```

```python
len(df)
```

```python
df.isnull().sum()
```

```python
df['MONTHID'].isnull().sum() / len(df)
```

# Avista Only

```python
%%time
avista_corr_spearman = df[COLUMN_GROUPS['outcome']+COLUMN_GROUPS['avista']+COLUMN_GROUPS['eng_pred']].corr(method='spearman')
```

```python
sns.set(font_scale=0.5)
sns.heatmap(data=abs(avista_corr_spearman), annot=True, fmt='.1f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.show();
```

```python
sns.set(font_scale=0.5)
sns.heatmap(data=abs(avista_corr_spearman.loc[COLUMN_GROUPS['avista']+COLUMN_GROUPS['eng_pred']][COLUMN_GROUPS['outcome']]), annot=True, fmt='.3f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.show();
```

# Drop Nulls

```python
%%time
drop_nulls_corr_spearman = df.drop(COLUMN_GROUPS['id']+COLUMN_GROUPS['problem'], axis=1).dropna().corr(method='spearman')
```

```python
sns.set(font_scale=0.5)
sns.heatmap(data=abs(drop_nulls_corr_spearman), annot=True, fmt='.1f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap']);
plt.show();
```

```python
sns.set(font_scale=0.5)
sns.heatmap(data=abs(drop_nulls_corr_spearman.loc[COLUMN_GROUPS['avista']+COLUMN_GROUPS['city']+COLUMN_GROUPS['combined']+COLUMN_GROUPS['eng_pred']][COLUMN_GROUPS['outcome']]), annot=True, fmt='.3f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap'])
plt.savefig(PLOT_SPECS['image_dest']+'corr_heatmap.png', bbox_inches='tight')
plt.show();
```

# Compare

```python
outcome = 'WITHIN_12_MO'

year_corr = drop_nulls_corr_spearman.loc[COLUMN_GROUPS['avista']+COLUMN_GROUPS['city']+COLUMN_GROUPS['combined']+COLUMN_GROUPS['eng_pred']][outcome].sort_values(ascending=False).to_frame()

year_corr.columns = ['Drop Nulls']

year_corr['Avista Only'] = avista_corr_spearman.loc[COLUMN_GROUPS['avista']+COLUMN_GROUPS['eng_pred']][outcome]

print("Percent of columns where corr Drop Nulls > corr Avista Only", (year_corr['Drop Nulls'] > year_corr['Avista Only']).sum() / len(year_corr))
print()
print(year_corr)
print()
print(year_corr.to_latex(float_format='%0.4f'))
```

## Conclusions


* Use the city data and drop nulls
    * Correlation was stronger between the predictors and COLUMN_GROUPS['outcome'] here than when only using Avista data


# Save

```python
df.dropna(subset=['MONTHID']).to_csv(PLOT_SPECS['data_source']+'data_02.csv', index=False)
```

<!-- #region tags=[] -->
# Time
<!-- #endregion -->

```python
print(time() - start)
```
