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

# Load Data

```python
from plot_config import *

start = time()
```

```python
def convert_arrearsmonth(date: int) -> pd.Timestamp:
    """Convert original arrearsmonth date format (YYYYMM) to a pandas Timestamp.

    Args:
        date (int): Date in original format (YYYYMM).

    Returns:
        (pandas.Timestamp): Date in converted format.
    """
    date_s = str(date)
    return pd.Timestamp(date_s[:4] + '-' + date_s[4:])


def load_data(data_path: str) -> (pd.DataFrame, pd.DataFrame):
    """Loads the billing and service agreements data.

    Args:
        data_path (str): Path to the data.

    Returns:
        (pd.DataFrame, pd.DataFrame): The billing dataframe and the service agreements dataframe.
    """
    # Billing Data
    file_years = ['2015', '2016', '2017', '2018', '2019', '2020']
    path = '/SpaData_'
    bill = pd.DataFrame()
    for file_year in file_years:
        bill = pd.concat([bill, pd.read_csv(data_path+path+file_year+'_Anon.csv')])

    bill['ARREARSMONTH'] = bill['ARREARSMONTH'].apply(convert_arrearsmonth)
    bill = bill.rename({'ARREARSMONTH': 'MONTH'}, axis=1)

    dtypes = {
        'MONTH': 'datetime64',
        'RES_EL_CUR120_DAYS': 'float',
        'RES_EL_CUR22_DAYS': 'float',
        'RES_EL_CUR30_DAYS': 'float',
        'RES_EL_CUR60_DAYS': 'float',
        'RES_EL_CUR90_DAYS': 'float',
        'RES_EL_CUR_BAL_AMT': 'float',
        'RES_EL_OVER_120_DAYS': 'float',
        'RES_GAS_CUR120_DAYS': 'float',
        'RES_GAS_CUR22_DAYS': 'float',
        'RES_GAS_CUR30_DAYS': 'float',
        'RES_GAS_CUR60_DAYS': 'float',
        'RES_GAS_CUR90_DAYS': 'float',
        'RES_GAS_CUR_BAL_AMT': 'float',
        'RES_GAS_OVER_120_DAYS': 'float',
        'BREAK_ARRANGEMENT': 'float',
        'BREAK_PAY_PLAN': 'float',
        'CALL_OUT': 'float',
        'CALL_OUT_MANUAL': 'float',
        'DUE_DATE': 'float',
        'FINAL_NOTICE': 'float',
        'PAST_DUE': 'float',
        'SEVERANCE_ELECTRIC': 'float',
        'SEVERANCE_GAS': 'float',
        'MONTHID': 'float',
        'CITY_TOT_DUE': 'float',
        'CITY_30_DAYS_PAST_DUE_AMT': 'float',
        'CITY_60_DAYS_PAST_DUE_AMT': 'float',
        'CITY_90_DAYS_PAST_DUE_AMT': 'float',
        'SPA_PREM_ID': 'int',
        'SPA_ACCT_ID': 'int',
        'COVID_REMINDER': 'float'
    }
    for col in dtypes:
        bill[col] = bill[col].astype(dtypes[col])

    # Service Agreements
    sa = pd.read_csv(data_path+'/ServiceAgreements_Anon.csv').rename({
        'spa_prem_id': 'SPA_PREM_ID',
        'spa_acct_id': 'SPA_ACCT_ID',
        'spa_sa_id': 'SPA_SA_ID',
        'spa_per_id': 'SPA_PER_ID',
        'homelessMatch': 'CMIS_MATCH',
        'Class': 'CLASS',
        'apartment': 'APARTMENT',
        'EnrollDate': 'ENROLL_DATE'
    }, axis=1)

    dtypes = {
        'SPA_PREM_ID': 'int',
        'SPA_ACCT_ID': 'int',
        'SPA_SA_ID': 'int',
        'SPA_PER_ID': 'int',
        'ACCT_REL_TYPE_CD': 'str',
        'CMIS_MATCH': 'object',
        'START_DT': 'datetime64',
        'END_DT': 'datetime64',
        'SA_TYPE_DESCR': 'str',
        'CLASS': 'str',
        'APARTMENT': 'bool',
        'ENROLL_DATE': 'datetime64'
    }
    for col in dtypes:
        sa[col] = sa[col].astype(dtypes[col])

    return bill, sa
```

```python
def print_stats(df: pd.DataFrame, group: bool=False) -> None:
    indexes = {
        0: ['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH'],
        1: ['SPA_ACCT_ID', 'SPA_PREM_ID'],
        2: ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']
    }
    cols = set(df.columns.tolist())
    choice = 0
    if set(indexes[2]).issubset(cols):
        choice = 2
    elif set(indexes[0]).issubset(cols):
        choice = 0
    elif set(indexes[1]).issubset(cols):
        choice = 1
    else:
        print('Index not recognized.')
    
    for col in indexes[choice]:
        print(f'{df[col].nunique()} unique {col}s')
    if group:
        other_cols = [col for col in df.columns if col not in indexes[choice]]
        print(f'\nMax number of unique values per {indexes[choice]}')
        print(df.groupby(indexes[choice])[other_cols].nunique().max())
    return
```

```python
bill, sa = load_data(PLOT_SPECS['data_source'])
```

```python
print(len(bill))
print(len(sa))
```

```python
100 * 3126613 / len(bill)
```

```python
print_stats(df=sa, group=True)
```

```python
%%time
print_stats(df=bill, group=True)
```

# Validate Billing Data

```python
bill.dtypes
```

```python
bill.isnull().sum()
```

## Examine Columns other than Amounts Owed

```python
cols = [
    # 'MONTH',
    'BREAK_ARRANGEMENT',
    'BREAK_PAY_PLAN',
    'CALL_OUT',
    'CALL_OUT_MANUAL',
    'DUE_DATE',
    'FINAL_NOTICE',
    'PAST_DUE',
    'SEVERANCE_ELECTRIC',
    'SEVERANCE_GAS',
    'MONTHID',
    'SPA_PREM_ID',
    'SPA_ACCT_ID',
    'COVID_REMINDER'
]
for col in cols:
    vals = bill[col].unique()
    if len(vals) < 50:
        print(f'{col}: {sorted(vals)}')
    else:
        print(f'{col}: {vals}')
```

### Counts: Replace 'nan' with 0, convert to integer

```python
cols = [col for col in cols if col not in ['MONTHID', 'SPA_PREM_ID', 'SPA_ACCT_ID']]
for col in cols:
    bill[col] = bill[col].replace(to_replace=np.nan, value=0).astype('int')
```

```python
bill.isnull().sum()
```

Deal with these later.


## Establish Composite Key
'SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH'

```python
groupers = ['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']
cols = [col for col in bill.columns.tolist() if col not in groupers]
```

```python
bill.groupby(groupers)[cols].nunique(dropna=False).max()
```

Are these duplicates due to null values?

```python
bill.groupby(groupers)[cols].nunique(dropna=True).max()
```

Duplicates are not entirely from null values.  
They must be from the matching of City data to Avista data - how many duplicates are there?

```python
dups = bill.groupby(groupers)['MONTHID'].nunique()
dups = dups[dups > 1].reset_index().drop('MONTH', axis=1).drop_duplicates()
dups['Combined'] = dups.apply(lambda x: (x['SPA_ACCT_ID'], x['SPA_PREM_ID']), axis=1)
print(f"{len(dups)} = {100 * len(dups) / len(bill[['SPA_ACCT_ID', 'SPA_PREM_ID']].drop_duplicates()):0.2f}% of unique (SPA_ACCT_ID, SPA_PREM_ID) pairs.")
```

```python
bill = bill.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
dups1 = bill[bill.index.isin(dups['Combined'])]
bill = bill.reset_index()
city_cols = [
    'MONTH',
    'MONTHID',
    'CITY_TOT_DUE',
    'CITY_30_DAYS_PAST_DUE_AMT',
    'CITY_60_DAYS_PAST_DUE_AMT',
    'CITY_90_DAYS_PAST_DUE_AMT'
]
dups1[city_cols].drop_duplicates().head()
```

No guidance on which to keep - drop all duplicates.


### Duplicates: Drop All

```python
prev = len(bill)
bill = bill.set_index(groupers)
dups = bill.copy()
dups = dups[dups.groupby(groupers).size() > 1].index
bill = bill[~bill.index.isin(dups)]
bill = bill.reset_index()
print(f'Lost {prev-len(bill)} rows.')
```

```python
# Check Unique
bill.groupby(groupers)[cols].nunique().max()
```

## Billing Data Ready to Match 
Matching on 'SPA_ACCT_ID' and 'SPA_PREM_ID'.


# Validate Service Agreements

```python
sa.dtypes
```

```python
sa.isnull().sum()
```

## Keep only the 'MAIN' acct holder
Assume the 'MAIN' account holder pays all the bills.

```python
# Retain only 'MAIN' account holders
prev = len(sa)
sa = sa[sa['ACCT_REL_TYPE_CD'] == 'MAIN']
print(f'Lost {prev - len(sa)} rows.')
```

```python
sa.groupby('SPA_PER_ID')['SPA_ACCT_ID'].nunique().max()
```

```python
sa.groupby('SPA_ACCT_ID')['SPA_PER_ID'].nunique().max()
```

One person can have multiple accounts but a single account can only be associated with one person (after dropping everyone except the MAIN account holder).


## Validate outcome
* 'CMIS_MATCH': Replace NaN with False - assume these people have not experienced homelessness.
* 'CMIS_MATCH': If single True for person, then set all as True for person.

```python
sa['CMIS_MATCH'] = sa['CMIS_MATCH'].replace(to_replace=np.nan, value=False).astype('bool')
```

```python
dups = sa.groupby('SPA_PER_ID')['CMIS_MATCH'].nunique()
dups = dups[dups > 1]
print(f'Number of people with non-unique outcomes: {dups.index.nunique()} = {100 * dups.index.nunique() / sa["SPA_PER_ID"].nunique():0.2f}%')
dups = sa[sa['SPA_PER_ID'].isin(dups.index)].set_index(['SPA_PER_ID', 'SPA_ACCT_ID', 'SPA_SA_ID']).sort_index()
dups.head(20)
```

```python
dups = sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID'])['CMIS_MATCH'].nunique()
dups = dups[dups > 1].reset_index()
dups = set(dups.apply(lambda x: (x['SPA_ACCT_ID'], x['SPA_PREM_ID']), axis=1).values)
print(f'Number of ACCT-PREMs with non-unique outcomes: {len(dups)} ={100 * len(dups) / len(sa[["SPA_ACCT_ID", "SPA_PREM_ID"]].drop_duplicates()): 0.2f}%')
dups = sa[sa.apply(lambda x: (x['SPA_ACCT_ID'], x['SPA_PREM_ID']) in dups, axis=1)].drop_duplicates().set_index(['SPA_ACCT_ID', 'SPA_PREM_ID']).sort_index()
dups.head(20)
```

It appears the non-unique outcome measure per 'SPA_PER_ID' is due to some matching errors between accounts and people. The best we can do is hope the positive labels are correct and set 'CMIS_MATCH' = TRUE for all instances of a person with any 'CMIS_MATCH' = TRUE.

```python
sa = sa.set_index('SPA_PER_ID')
prev = (sa['CMIS_MATCH'] == True).sum()
sa.update(sa.groupby('SPA_PER_ID')['CMIS_MATCH'].any())
print(f"Added {(sa['CMIS_MATCH'] == True).sum() - prev} True values to 'CMIS_MATCH'.")
prev = len(sa)
sa = sa.reset_index().drop_duplicates()
print(f'Removed {prev - len(sa)} duplicate rows.')
```

## Examine Duplicates

```python
groupers = ['SPA_ACCT_ID', 'SPA_PREM_ID']
cols = [col for col in sa.columns if col not in groupers]

# Check Unique
sa.groupby(groupers)[cols].nunique().max()
```

### Retain earliest 'ENROLL_DATE'

```python
sa = sa.set_index('SPA_PER_ID')
sa.update(sa.groupby('SPA_PER_ID')['ENROLL_DATE'].min())
sa = sa.reset_index()
```

```python
sa.groupby('SPA_PER_ID')['ENROLL_DATE'].nunique().max()
```

### Retain subset of columns

```python
sa = sa[['SPA_ACCT_ID', 'SPA_PREM_ID', 'SPA_PER_ID', 'ENROLL_DATE', 'CMIS_MATCH']].drop_duplicates()
```

```python
sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID']).nunique().max()
```

## Service Agreement Data Ready to Match 
Matching on 'SPA_ACCT_ID' and 'SPA_PREM_ID'.


# Join Billing and Service Agreements

```python
bill = bill.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])

df = bill.join(sa, how='inner')
df = df.reset_index()
bill = bill.reset_index()
sa = sa.reset_index()
print(f'Lost {len(bill) - len(df)} rows.')
```

## Check Nulls

```python
df.isnull().sum()
```

Investigate Nulls further in 02_Data_Exploration.ipynb


## Check Outcome

```python
df[df['CMIS_MATCH']]['ENROLL_DATE'].isnull().sum()
```

```python
df[df['ENROLL_DATE'].notnull()]['CMIS_MATCH'].value_counts()
```

* All people that were recorded as experiencing homelessness have an 'ENROLL_DATE'.
* All people that have an 'ENROLL_DATE' also have 'CMIS_MATCH' == TRUE.
* This is what we want.


## Check Uniqueness

```python
groupers = ['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']
cols = [col for col in df.columns if col not in groupers]

# Check Unique
df.groupby(groupers)[cols].nunique(dropna=False).max()
```

```python
groupers = ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']
cols = [col for col in df.columns if col not in groupers]

# Check Unique
df.groupby(groupers)[cols].nunique(dropna=False).max()
```

```python
# Number of rows effected
dups = df.groupby(groupers)['SPA_ACCT_ID'].nunique(dropna=False).copy()
dups = dups[dups > 1]
print(f"{len(dups)} = {100 * len(dups) / df['SPA_ACCT_ID'].nunique():0.2f}")
```

```python
%%time
cols = [
    'SPA_ACCT_ID',
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
    'BREAK_ARRANGEMENT',
    'BREAK_PAY_PLAN',
    'CALL_OUT',
    'CALL_OUT_MANUAL',
    'DUE_DATE',
    'FINAL_NOTICE',
    'PAST_DUE',
    'SEVERANCE_ELECTRIC',
    'SEVERANCE_GAS',
]
df = df.set_index(groupers).sort_index()
dups_plus_minus = pd.DataFrame(columns=groupers+cols)
delta = pd.offsets.DateOffset(months=1)
for dup in dups.index:
    center = dup[2]
    for day in [center + x * delta for x in range(-3, 3, 1)]:
        idx = (dup[0], dup[1], day)
        if idx in df.index:
            dups_plus_minus = pd.concat([
                dups_plus_minus,
                df.loc[idx][cols].reset_index()
            ], axis=0)

dups_plus_minus = dups_plus_minus.set_index(groupers+['SPA_ACCT_ID']).sort_index()
df = df.reset_index()
```

```python
dups_plus_minus.head(15)
```

```python
dups_plus_minus.loc[(13325, 85262, pd.Timestamp('2018-11-01'))].values
```

```python
# People Effected
dup_ids = set(dups.index.get_level_values('SPA_PER_ID'))
print(len(dup_ids))
df[df['SPA_PER_ID'].isin(dup_ids) & df['CMIS_MATCH']]['SPA_PER_ID'].nunique()
```

Most duplicates are from the same person having two accounts at the same premises and month. This likely occurs when someone switches types of accounts since the overlap often only occurs for one month. Other duplicates appear to just be duplicated information.   
Duplication only effects 164 keys (75 people) and all are negatives cases. Drop the duplicates.

```python
# Remove duplicate rows
df = df.set_index(groupers)
df = df[~df.index.isin(dups.index)]
df = df.reset_index()
```

```python
groupers = ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']
cols = [col for col in df.columns if col not in groupers]

# Check Unique
df.groupby(groupers)[cols].nunique(dropna=False).max()
```

## Data Coverage

```python
# People Negatives vs Positives
df.groupby('CMIS_MATCH')['SPA_PER_ID'].nunique()
```

```python
# Rows per year
df['MONTH'].dt.year.value_counts()
```

# Create Additional Features


### Predictors

```python
# Grouped Avista amounts owed
cols_el = [
    'RES_EL_CUR120_DAYS',
    'RES_EL_CUR22_DAYS',
    'RES_EL_CUR30_DAYS',
    'RES_EL_CUR60_DAYS',
    'RES_EL_CUR90_DAYS',
    'RES_EL_CUR_BAL_AMT',
    'RES_EL_OVER_120_DAYS'
]
cols_gas = [
    'RES_GAS_CUR120_DAYS',
    'RES_GAS_CUR22_DAYS',
    'RES_GAS_CUR30_DAYS',
    'RES_GAS_CUR60_DAYS',
    'RES_GAS_CUR90_DAYS',
    'RES_GAS_CUR_BAL_AMT',
    'RES_GAS_OVER_120_DAYS',
]
for i in range(len(cols_el)):
    s = cols_el[i][6:]
    df['AVISTA'+s] = df[cols_el[i]] + df[cols_gas[i]]
```

```python
# Overall amounts owed
cols_el = [
    'RES_EL_CUR_BAL_AMT',
    'RES_EL_CUR30_DAYS',
    'RES_EL_CUR60_DAYS',
    'RES_EL_CUR90_DAYS'
]
cols_gas = [
    'RES_GAS_CUR_BAL_AMT',
    'RES_GAS_CUR30_DAYS',
    'RES_GAS_CUR60_DAYS',
    'RES_GAS_CUR90_DAYS'
]
cols_city = [
    'CITY_TOT_DUE',
    'CITY_30_DAYS_PAST_DUE_AMT',
    'CITY_60_DAYS_PAST_DUE_AMT',
    'CITY_90_DAYS_PAST_DUE_AMT',
]
for i in range(len(cols_el)):
    s = cols_el[i][6:]
    df['TOTAL'+s] = df[cols_el[i]] + df[cols_gas[i]] + df[cols_city[i]]
```

```python
def accumulate(df: pd.DataFrame, grp_by_col: str, cumulative_col: str, new_col_name: str,
               month_col: str = 'MONTH') -> pd.DataFrame:
    """
    Finds cumulative counts over time.
    """
    cumulative = df[[month_col, grp_by_col, cumulative_col]].copy()
    # Find number of unique cumulateive elements
    cumulative = cumulative.drop_duplicates([grp_by_col, cumulative_col], keep='first').groupby(
        [grp_by_col, month_col]).nunique()
    # Find cumulative count of unique elements
    cumulative[new_col_name] = (cumulative.groupby(grp_by_col)[cumulative_col].cumcount() + 1.0)
    cumulative.drop(cumulative_col, axis=1, inplace=True)
    # Join counts back to df
    new_df = df.join(cumulative, how='left', on=[grp_by_col, month_col])
    # Forward fill index gaps
    new_df[new_col_name] = new_df[new_col_name].ffill().astype('int64')

    return new_df
```

```python
# Create 'NUM_PREM_FOR_PER': The cumulative number of premesis a person is associated with over time.
df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')
```

```python
# Create 'NUM_PER_FOR_PREM': The total number of people seen by each premesis in our data.
num_per_for_prem = df.groupby('SPA_PREM_ID')['SPA_PER_ID'].nunique()
num_per_for_prem.name = 'NUM_PER_FOR_PREM'
df = df.set_index('SPA_PREM_ID')
df['NUM_PER_FOR_PREM'] = num_per_for_prem
df = df.reset_index()
```

<!-- #region tags=[] -->
### Outcomes
<!-- #endregion -->

```python
person_enrollments = df[['SPA_PER_ID', 'ENROLL_DATE']].drop_duplicates().set_index('SPA_PER_ID').squeeze()

df['MO_AWAY'] = df.apply(lambda x: person_enrollments.loc[x['SPA_PER_ID']] - x['MONTH'], axis=1)
```

```python
# Drop negative values (months after people have experienced homelessness)
df = df[~(df['MO_AWAY'] < np.timedelta64(0, 'D'))]

# Create additional outcome measures
df['WITHIN_12_MO'] = df['MO_AWAY'] <= np.timedelta64(12, 'M')
df['WITHIN_6_MO'] = df['MO_AWAY'] <= np.timedelta64(6, 'M')
df['WITHIN_3_MO'] = df['MO_AWAY'] <= np.timedelta64(3, 'M')
df['WITHIN_1_MO'] = df['MO_AWAY'] <= np.timedelta64(1, 'M')

df['MO_AWAY'] = df['MO_AWAY'].dt.days / 30.0
```

<!-- #region tags=[] -->
# Save
<!-- #endregion -->

```python
df.to_csv(PLOT_SPECS['data_source']+'data_01.csv', index=False)
```

<!-- #region tags=[] -->
# Time
<!-- #endregion -->

```python
print(time() - start)
```
