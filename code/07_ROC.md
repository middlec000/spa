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

```python
from plot_config import *
from sklearn.metrics import roc_curve, auc
start_time = time()
```

```python
df = pd.read_csv(PLOT_SPECS['data_source']+'predictions_log.csv')
df.head()
```

```python
def plot_roc(df: pd.DataFrame, filename: str):
    alpha = 0.3
    aucs = []
    # Get valid steps
    steps = set({})
    for col in df.columns:
        steps.add(int(col.split('_')[2]))
    
    for step in steps:
        fpr, tpr, thresholds = roc_curve(y_true=df['y_true_'+str(step)].dropna(), y_score=df['y_pred_'+str(step)].dropna())
        aucs.append(auc(fpr, tpr))
        if step == max(steps): # last step
            label = f'Mean, Std AUC = {np.mean(aucs):0.3f}, {np.std(aucs):0.3f}'
            sns.lineplot(x=fpr, y=tpr, color='black', alpha=alpha, label=label)
        else:
            sns.lineplot(x=fpr, y=tpr, color='black', alpha=alpha)
    
    plt.grid(which='major', color='grey')
    plt.grid(which='minor', color='lightgrey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.minorticks_on()
    plt.savefig(PLOT_SPECS['image_dest']+filename+'.png', bbox_inches='tight')
    plt.show()
```

```python
%%time
plot_roc(df=df, filename='roc_log')
```

```python
print(time() - start_time)
```
