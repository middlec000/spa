import papermill as pm

for notebook in [
    '01_Data-Validation',
    '02_City-Nulls',
    '03_Which-Years',
    # '04_Patterns-in-Time',
    '05_Variable-Selection',
    '06_Logistic_Statsmodels',
    '07_roc'
]:
    pm.execute_notebook(
        notebook+'.ipynb', # input notebook
        notebook+'.ipynb' # output notebook
    )
