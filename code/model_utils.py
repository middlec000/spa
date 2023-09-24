import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from copy import copy


def get_metrics(y_true: np.array, y_pred: np.array, thresh: float = 0.5) -> dict:
    metrics = {
        'auc': roc_auc_score(y_true=y_true, y_score=y_pred)
    }
    if y_pred.dtype == float:
        new = np.array([y >= thresh for y in y_pred])
    else:
        new = y_pred
    M = pd.DataFrame(data=confusion_matrix(y_true=y_true, y_pred=new), index=['f_act', 't_act'], columns=['f_pred', 't_pred'])
    metrics['confusion_matrix'] = M
    try:
        metrics['acc'] = (M.loc['t_act', 't_pred'] + M.loc['f_act', 'f_pred']) / M.sum().sum() # (tp + tn) / total
    except:
        metrics['acc'] = None
    try:
        metrics['tpr'] = M.loc['t_act', 't_pred'] / (M.loc['t_act', 't_pred'] + M.loc['t_act', 'f_pred']) # tp / (tp + fn)
    except:
        metrics['tpr'] = None
    try:
        metrics['fnr'] = 1.0 - metrics['tpr'] # 1 - TPR
    except:
        metrics['fnr'] = None
    try:
        metrics['fpr'] = M.loc['f_act', 't_pred'] / (M.loc['f_act', 't_pred'] + M.loc['f_act', 'f_pred']) # fp / (fp + tn)
    except:
        metrics['fpr'] = None
    try:
        metrics['tnr'] = 1.0 - metrics['fpr'] # 1 - FPR
    except:
        metrics['tnr'] = None
    try:
        metrics['ppv'] = M.loc['t_act', 't_pred'] / (M.loc['t_act', 't_pred'] + M.loc['f_act', 't_pred'])# tp / (tp + fp)
    except:
        metrics['ppv'] = None
    try:
        metrics['ba'] = (metrics['tpr'] + metrics['tnr']) / 2 # (TPR + TNR) / 2
    except:
        metrics['ba'] = None
    try:
        metrics['f1'] = 2 * (metrics['ppv'] * metrics['tpr']) / (metrics['ppv'] + metrics['tpr']) # 2 * (PPV * TPR) / (PPV + TPR)
    except:
        metrics['f1'] = None

    return metrics


def get_folds(n_folds: int, n_repeats: int, data: pd.DataFrame, x_cols: list, y_col: str, random_seed: int) -> tuple:
    """Create the indexes for Stratified Repeated K-Folds.
    
    The data is stratified on 'CMIS_MATCH', meaning that each fold contains roughly the same proportion of people with 'CMIS_MATCH' == True.
    """
    rng = np.random.default_rng(random_seed)
    df = data.set_index(['SPA_PER_ID', 'MONTH']).sort_index()
    X = df[x_cols].values
    y = df[y_col].values
    check = df['CMIS_MATCH'].values
    idx_ppl = df.index.get_level_values('SPA_PER_ID').values
    idx_mo = df.index.get_level_values('MONTH').values
    pos_ppl = df[df['CMIS_MATCH']].index.get_level_values('SPA_PER_ID').unique().values
    neg_ppl = df[~df['CMIS_MATCH']].index.get_level_values('SPA_PER_ID').unique().values
    
    # Create map from person to list of data indices
    per_to_idx = {}
    prev_per = None
    for i in range(len(idx_ppl)):
        cur_per = idx_ppl[i]
        if cur_per == prev_per: 
            per_to_idx[cur_per] += [i]
        else:
            per_to_idx[cur_per] = [i]
        prev_per = cur_per
    
    n_pos_per_fold = int(np.floor(len(pos_ppl) / n_folds))
    n_neg_per_fold = int(np.floor(len(neg_ppl) / n_folds))

    repeats = []
    for j in range(n_repeats):
        # Set up people to pull from for this repeat
        pos_ppl_repeat = copy(pos_ppl)
        neg_ppl_repeat = copy(neg_ppl)
        rng.shuffle(pos_ppl_repeat)
        rng.shuffle(neg_ppl_repeat)

        # Get the folds for this repeat
        folds = []
        for i in range(1, n_folds+1):
            if i == n_folds: # if last fold, take remaining people
                pos_ppl_draw = pos_ppl_repeat[(i-1)*n_pos_per_fold:]
                neg_ppl_draw = neg_ppl_repeat[(i-1)*n_neg_per_fold:]
            else:
                pos_ppl_draw = pos_ppl_repeat[(i-1)*n_pos_per_fold: i*n_pos_per_fold]
                neg_ppl_draw = neg_ppl_repeat[(i-1)*n_neg_per_fold: i*n_neg_per_fold]
            ppl_draw = np.concatenate([pos_ppl_draw, neg_ppl_draw])
            # Get fold index
            fold_idx = []
            for per in ppl_draw:
                fold_idx += per_to_idx[per]
            folds.append(fold_idx)
            
        repeats.append(folds)
    
    return X, y, repeats, check
