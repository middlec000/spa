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

# Investigate Patterns over Time


## Intuition
We hypothesis that as a person get closer to experiencing homelessness (as 'MO_AWAY' $\to0$), they display more signs of financial distress and become more distinguishable from the general population.
## Strategy
Plot key variables against 'MO_AWAY' for positive cases and analyze if they are distinguishable from the negative cases.

<!-- #region pycharm={"name": "#%% md\n"} tags=[] -->
# Setup
<!-- #endregion -->

```python hidden=true pycharm={"name": "#%%\n"}
from plot_config import *

start_time = time()
```

```python
df = pd.read_csv(PLOT_SPECS['data_source']+'data_03.csv', low_memory=False)
for col in ['MONTH', 'ENROLL_DATE']:
    df[col] = df[col].astype('datetime64')
```

```python pycharm={"name": "#%%\n"}
pos = df[df['CMIS_MATCH']]
neg = df[~df['CMIS_MATCH']]
len(pos[pos['MO_AWAY'].isnull()])
```

```python
XMIN = pos['MO_AWAY'].min()
XMAX = pos['MO_AWAY'].max()
```

## Plotting Function

```python
def plot_over_time(pos: pd.DataFrame, neg: pd.DataFrame, on: str, ylim: tuple[int], save: bool=False) -> None:
    # Plot P
    sns.kdeplot(data=pos, x="MO_AWAY", y=on, color=PLOT_SPECS['cmap'].colors[0], fill=True, label='P: mean +/- std')

    # Plot N mean +/- std
    neg_mean = neg[on].describe()['mean']
    neg_std = neg[on].describe()['std']

    plt.hlines(mean, XMIN=XMIN, XMAX=XMAX, color='black', label='N: mean', linestyle='dashdot')
    plt.hlines([neg_mean+neg_std, neg_mean-neg_std], XMIN=XMIN, XMAX=XMAX, color='black', label='N: mean +/- std', linestyle='dashed')

    # Set labels
    plt.ylabel(on, fontsize=PLOT_SPECS['fontsize'])
    plt.yticks(fontsize=PLOT_SPECS['fontsize']-2)
    plt.ylim(ylim)
    plt.xlabel('MO_AWAY', fontsize=PLOT_SPECS['fontsize'])
    plt.xticks(fontsize=PLOT_SPECS['fontsize']-2)
    plt.legend(fontsize=PLOT_SPECS['fontsize'], loc='upper right')

    if save:
        plt.savefig(PLOT_SPECS['image_dest']+'PN_mo-away_on_'+on+'.png', dpi=300, bbox_inches='tight')
    plt.show();
    return None
```

# Plots


## PAST_DUE

```python
plot_over_time(pos=pos, neg=neg, on='PAST_DUE', ylim=(-2, 12))
```

## TOTAL_CUR60_DAYS

```python
plot_over_time(pos=pos, neg=neg, on='TOTAL_CUR60_DAYS', ylim=(-150, 800))
```

## NUM_PER_FOR_PREM

```python
plot_over_time(pos=pos, neg=neg, on='NUM_PER_FOR_PREM', ylim=(0, 7))
```

## NUM_PREM_FOR_PER

```python
plot_over_time(pos=pos, neg=neg, on='NUM_PREM_FOR_PER', ylim=(0.5, 2.2))
```

## CALL_OUT

```python
plot_over_time(pos=pos, neg=neg, on='CALL_OUT', ylim=(-2, 8.5))
```

## DUE_DATE

```python
plot_over_time(pos=pos, neg=neg, on='DUE_DATE', ylim=(-1.5, 7.5))
```

## FINAL_NOTICE

```python
plot_over_time(pos=pos, neg=neg, on='FINAL_NOTICE', ylim=(-2, 10))
```

## SEVERANCE_ELECTRIC

```python
plot_over_time(pos=pos, neg=neg, on='SEVERANCE_ELECTRIC', ylim=(-1.5, 8))
```

# Conclusion


'PAST_DUE', 'NUM_PER_FOR_PREM', and 'FINAL_NOTICE' showed some distinction between P and N, especially as 'MO_AWAY' approached 0.  
Otherwise the variables showed little to no distinction between P and N.


# How long from last bill to homelessness?

```python
data = pd.concat([
    pd.read_csv(PLOT_SPECS['data_source']+'data_03.csv', low_memory=False), 
    pd.read_csv(PLOT_SPECS['data_source']+'data_03_2020.csv', low_memory=False)], 
    axis=0
)

for col in ['MONTH', 'ENROLL_DATE']:
    data[col] = data[col].astype('datetime64')
```

```python
data.columns.tolist()
```

```python
# For each person, find last billing month and enroll data, then diff
gap = data.groupby('SPA_PER_ID')['MONTH'].max()
gap = gap.to_frame().join(data.groupby('SPA_PER_ID')['ENROLL_DATE'].min(), how='left')
gap.head()
```

```python
gap['dist'] = gap['ENROLL_DATE'] - gap['MONTH']
```

```python
gap[gap['dist'].notnull()].head()
```

```python
gap[gap['dist'].notnull()]['dist'].dt.days.hist()
```

```python
gap[gap['dist'].notnull()]['dist'].dt.days.describe()
```

```python
table = (gap[gap['dist'].notnull()]['dist'].dt.days / 365.0).describe(percentiles=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0])
# print(table)
print(table.loc[['5%', '10%', '20%', '50%', '80%', '90%', '95%', '100%']].to_frame().transpose().to_latex(float_format='%0.3f'))
```

```python
type(table)
```

# Earliest billing records

```python
early = data.groupby('SPA_PER_ID')['MONTH'].min()
early = early.to_frame().join(data[['SPA_PER_ID', 'CMIS_MATCH']].drop_duplicates().set_index('SPA_PER_ID'), how='left')
early.head()
```

```python
early['MONTH'].hist()
```

```python
early[early['CMIS_MATCH']]['MONTH'].hist()
```

```python
for year in range(2016, 2021, 1):
    print(year, (early[early['CMIS_MATCH']]['MONTH'] > pd.Timestamp(f'{year}-01-01')).sum())
```

# Multiple Locations for Same Month

```python
df.columns.to_list()
```

```python
dups = df.groupby(['SPA_PER_ID', 'MONTH'])['SPA_PREM_ID'].nunique()
dups.max()
```

```python
dups.describe(percentiles=[0.80, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999])
```

```python
dups.head()
```

```python
dups = dups.groupby('SPA_PER_ID').max()
dups.head()
```

```python
dups = dups.to_frame().join(df.groupby('SPA_PER_ID')['CMIS_MATCH'].any(), how='left')
dups.head()
```

```python
dups.corr(method='spearman')
```

# Time

```python
print(time() - start_time)
```
