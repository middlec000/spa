import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


# Set plot size and quality
plt.rcParams.update({'figure.figsize': (11,7), 'figure.dpi': 120})

# Other plotting parameters to pass to seaborn or matplotlib
PLOT_SPECS = {
    'fontsize': 14, 
    'palette': 'Greys',
    'saturation': 0.5,
    'image_dest': 'results_images/',
    'data_source': '../data/'
}

sns.set_palette(sns.color_palette(PLOT_SPECS['palette']))

PLOT_SPECS['cmap'] = ListedColormap(sns.color_palette().as_hex())

COLUMN_GROUPS = {
    'id': [
        'SPA_PREM_ID',
        'SPA_PER_ID',
        'MONTH',
        'SPA_ACCT_ID',
        'MONTHID',
    ],
    'avista': [
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
        'COVID_REMINDER',
    ],
    'avista_agg': [
        'AVISTA_CUR120_DAYS',
        'AVISTA_CUR22_DAYS',
        'AVISTA_CUR30_DAYS',
        'AVISTA_CUR60_DAYS',
        'AVISTA_CUR90_DAYS',
        'AVISTA_CUR_BAL_AMT',
        'AVISTA_OVER_120_DAYS',
    ],
    'city': [
        'CITY_TOT_DUE',
        'CITY_30_DAYS_PAST_DUE_AMT',
        'CITY_60_DAYS_PAST_DUE_AMT',
        'CITY_90_DAYS_PAST_DUE_AMT',
    ],
    'combined': [
        'TOTAL_CUR_BAL_AMT',
        'TOTAL_CUR30_DAYS',
        'TOTAL_CUR60_DAYS',
        'TOTAL_CUR90_DAYS',
    ],
    'eng_pred': [
        'NUM_PREM_FOR_PER',
        'NUM_PER_FOR_PREM'
    ],
    'outcome': [
        'CMIS_MATCH',
        'WITHIN_1_MO',
        'WITHIN_3_MO',
        'WITHIN_6_MO',
        'WITHIN_12_MO'
    ],
    'problem': [
        'ENROLL_DATE',
        'MO_AWAY'
    ]
}
