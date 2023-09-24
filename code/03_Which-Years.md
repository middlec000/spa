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

# Are Patterns in the Data Different in Different Years?


## Reasoning
* COVID-19 and the lifestyle changes it caused may have had an impact on how variables are related within the 2020 data
## Strategy
### Are the Outcomes Different in Different Years?
* Check if counts and percentages of positive/negatives are different from year to year
### Are Variables Related Differently During Different Years?
* Measure association between each predictor and outcome for each year  
* Compare rankings of associations from year to year

<!-- #region pycharm={"name": "#%% md\n"} tags=[] -->
# Setup
<!-- #endregion -->

```python hidden=true pycharm={"name": "#%%\n"}
from plot_config import *

start = time()
```

```python
df = pd.read_csv(PLOT_SPECS['data_source']+'data_02.csv', low_memory=False)
for col in ['MONTH', 'ENROLL_DATE']:
    df[col] = df[col].astype('datetime64')
```

```python
df[df['ENROLL_DATE'].notnull()]['CMIS_MATCH'].value_counts()
```

```python
df[df['CMIS_MATCH']]['ENROLL_DATE'].isnull().sum()
```

```python
df['ENROLL_DATE'].min()
```

```python
df['ENROLL_DATE'].max()
```

<!-- #region heading_collapsed=true hidden=true pycharm={"name": "#%% md\n"} tags=[] -->
# Outcome Events from Year to Year
<!-- #endregion -->

## Absolute Numbers

```python
len(df[df['ENROLL_DATE'].notnull()])
```

```python hidden=true pycharm={"name": "#%%\n"}
events_by_time = df[['SPA_PER_ID', 'ENROLL_DATE']].drop_duplicates().copy()
events_by_time.head()
```

```python
events_by_year = df[['SPA_PER_ID', 'ENROLL_DATE']].drop_duplicates().dropna().copy()
events_by_year['Year'] = events_by_year['ENROLL_DATE'].dt.year
events_by_year = events_by_year.groupby('Year')['SPA_PER_ID'].nunique().to_frame()
events_by_year = events_by_year.rename(columns={'SPA_PER_ID': 'Positives'})

people_by_year = df[['SPA_PER_ID', 'MONTH']].drop_duplicates().copy()
people_by_year['Year'] = people_by_year['MONTH'].dt.year
events_by_year['People'] = people_by_year.groupby('Year')['SPA_PER_ID'].nunique()
events_by_year['Proportion'] = events_by_year['Positives'].divide(events_by_year['People'])

events_by_year
```

```python
sns.barplot(x=events_by_year.index, y=events_by_year['Positives'], color='grey', alpha=0.7)
plt.ylabel('Number of People Experiencing First-time Homelessness')
plt.savefig(PLOT_SPECS['image_dest']+'num_hmls.png', bbox_inches='tight');
```

```python
sns.barplot(x=events_by_year.index, y=events_by_year['Proportion'], color='grey', alpha=0.7)
plt.ylabel('Proportion of People Experiencing First-time Homelessness')
plt.savefig(PLOT_SPECS['image_dest']+'prop_hmls.png', bbox_inches='tight');
```

<!-- #region hidden=true pycharm={"name": "#%% md\n"} -->
# Ranks of Correlation Between Variables from Year to Year
<!-- #endregion -->

```python hidden=true pycharm={"name": "#%%\n"}
%%time
outcome = 'WITHIN_12_MO'
keep = [outcome]+COLUMN_GROUPS['avista']+COLUMN_GROUPS['city']+COLUMN_GROUPS['combined']+COLUMN_GROUPS['eng_pred']

corr_year = {}
for year in range(2015, 2021):
    corr_year[year] = df[df['MONTH'].dt.year == year][keep].corr(method='spearman')[outcome].drop(outcome, axis=0)
```

```python
corr_years = pd.DataFrame(columns=list(range(2015,2021)))
for year in corr_years.columns:
    corr_years[year] = corr_year[year]
corr_years = corr_years.dropna()
corr_corr_years = corr_years.corr(method='spearman')
corr_corr_years
```

```python
sns.heatmap(data=corr_corr_years, annot=True, fmt='.3f', linewidths=0.01, linecolor='black', cmap=PLOT_SPECS['cmap'])
plt.savefig(fname=PLOT_SPECS['image_dest']+'corr_corr_year'+'.png', bbox_inches='tight')
plt.show();
```

## Conclusion


As hypothesized, the variables appear to be related to the outcome in different ways (at least different importances) in 2020 while the previous years are relatively similar.  

There are two more groups of years, [2015, 2016] and [2017, 2018, 2019], that are very similar within groups, but less similar between groups. However, the difference between these groups is much smaller than the difference between 2020 and the other years.


# Save

```python
df[~(df['MONTH'].dt.year == 2020)].to_csv(PLOT_SPECS['data_source']+'data_03.csv', index=False)
df[df['MONTH'].dt.year == 2020].to_csv(PLOT_SPECS['data_source']+'data_03_2020.csv', index=False)
```

# Time

```python
print(time() - start)
```
