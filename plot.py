import gpytorch
import torch
import os
#import math
#import tqdm
import urllib.request
from scipy.io import loadmat
from math import floor
import pandas as pd
import numpy as np
import time
import openpyxl
#from uci_datasets import Dataset
from plotnine import *
import plot_utils as pu
from pathlib import Path

path0 = os.getcwd() # or path to result_folder_dir
os.chdir(path0)
torch.manual_seed(20051210)
num_epochs = 400
skip = 50

def combined_metrics2(path):
    path1 = path
    file_names = [f for f in os.listdir(path1) if f != '.DS_Store'] #os.listdir(path1)
    for k in range(0,len(file_names)):
        ff = file_names[k]
        #files_only = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]
        #txt_files = [f for f in os.listdir(path1) if f.endswith('.txt') and os.path.isfile(os.path.join(path1, f))]
        #files = [f.name for f in Path(path1).iterdir() if f.is_file() and f.name.startswith('Met') and f.name.endswith('.txt')]
        files = [f.name for f in Path(path1,ff).iterdir() if f.is_file() and f.name.startswith('Met') and f.name.endswith('.txt')]
        print(sorted(files))
        dataframes = [torch.from_numpy(pd.read_csv(os.path.join(path1,ff,f)).to_numpy()) for f in sorted(files)]
        dat = torch.stack(dataframes,2)
        if num_epochs > skip:
            dat = dat[skip:,:]    
        mett = pd.DataFrame(torch.concat([dat.mean(0).t(),dat.std(0).t()]))
        mett.columns = ['rmse','mae','cove','msll','nlpd','perc','crps']
        nam = ff.removeprefix('result_')
        mett.to_csv(os.path.join(path0,'.'.join(['_'.join(['combine',nam]),'csv'])), index=False)
        clean_names = [os.path.splitext(f[17:])[0] for f in sorted(files)]
        with open(os.path.join(path0,'.'.join([nam,'txt'])), "w") as f:
            for name in clean_names:
                f.write(name + "\n")


combined_metrics2(path0)
#---------------------------------------plot
Metrics = ['RMSE','MAE','COVE','MSLL','NLL','PERC','CRPS']
old_list = [f.name for f in Path(path0).iterdir() if f.is_file() and f.name.startswith('') and f.name.endswith('.txt')]
new_list = ['DSPP-Chol','DSPP-MF','DGP-Chol*','DGP-MF*','PPGPR-OD','PPGPR-Chol','DGP-Chol','DGP-MF','ODSVGP','SVGP']
new_list3 = ['DSPP-Chol','DSPP-MF','DGP-Chol*','DGP-MF*','DGP-Chol','DGP-MF','ODSVGP','SVGP','PPGPR-OD','PPGPR-Chol']
#names_no_ext = [os.path.splitext(f)[0] for f in files]
#['DeepPredictiveLogLikelihood_DSPPHiddenLayer_CHOL_DSPP','DeepPredictiveLogLikelihood_DSPPHiddenLayer_MEANFLD_DSPP', 'PredictiveLogLikelihood_DGPHiddenLayer_CHOL_DGP','PredictiveLogLikelihood_DGPHiddenLayer_MEANFLD_DGP', 'PredictiveLogLikelihood_ODSVGPModel_GP', 'PredictiveLogLikelihood_SVGPModel_GP', 'VariationalELBO_DGPHiddenLayer_CHOL_DGP', 'VariationalELBO_DGPHiddenLayer_MEANFLD_DGP', 'VariationalELBO_ODSVGPModel_GP', 'VariationalELBO_SVGPModel_GP']
files = [f.name for f in Path(path0).iterdir() if f.is_file() and f.name.startswith('combine') and f.name.endswith('.csv')]
clean_names = [os.path.splitext(f)[0].replace("combine_", "", 1) for f in files]
#clean_names = [os.path.splitext(f[17:])[0] for f in files]
datas = [torch.from_numpy(pd.read_csv(os.path.join(path0,f)).to_numpy()) for f in files]
#datas = datas[:-1]
dat = torch.stack(datas,2).reshape(2,datas[0].shape[0]//2,datas[1].shape[1],len(datas))
dat2 = torch.stack([dat[0], dat[0] - dat[1], dat[0] + dat[1]],3)
#remove O-decoupled
ind = torch.tensor([1,4,9])
new_list2 = [item for i, item in enumerate(new_list) if i not in ind]
mask = torch.ones(dat2.size(0), dtype=torch.bool)
mask[ind] = False
dat22 = dat2[mask]  
ind2 = torch.tensor([0,1,2,3,6,7,8,9,4,5])
dat3 = dat22[ind2]

def assign_color(group):
    if "DSPP" in group:
        return "DSPP"
    elif "SVGP" in group or "PPGPR" in group:
        return "Single layer"
    elif "*" in group:
        return "This work"
    else:
        return "DGP"


#Generate latex tables
def make_table(df):
    df_sorted = df
    df_sorted['SD'] = df_sorted['Upper'] - df_sorted['Value']
    dfA = df_sorted[['Group','Point','Value',]]
    dfA = dfA.pivot_table(index='Group', columns='Point', values='Value', aggfunc='mean')
    dfB = df_sorted[['Group','Point','SD',]]
    dfB = dfB.pivot_table(index='Group', columns='Point', values='SD', aggfunc='mean')
    merged_df = dfB
    merged_df['3droad'] = round(dfA['3droad'],3).astype(str) + ' ± ' + round(dfB['3droad'],3).astype(str)
    merged_df['Bike'] = round(dfA['Bike'],3).astype(str) + ' ± ' + round(dfB['Bike'],3).astype(str)
    merged_df['Elevators'] = round(dfA['Elevators'],3).astype(str) + ' ± ' + round(dfB['Elevators'],3).astype(str)
    merged_df['Energy'] = round(dfA['Energy'],3).astype(str) + ' ± ' + round(dfB['Energy'],3).astype(str)
    merged_df['Jura'] = round(dfA['Jura'],3).astype(str) + ' ± ' + round(dfB['Jura'],3).astype(str)
    merged_df['Protein'] = round(dfA['Protein'],3).astype(str) + ' ± ' + round(dfB['Protein'],3).astype(str)
    #merged_df['3droad'] = round(dfA['3droad'],3).astype(str) + ' ± ' + round(dfB['3droad'],3).astype(str)
    latex_table = merged_df.to_latex(index=True, escape=False)
    latex_table = "\\small\n" + latex_table  # add \small for smaller font
    return latex_table
    #print(latex_table)
    #with open(os.path.join(path0,'.'.join([namm,'txt'])) as f:
        #f.write(Table)

dat4 = dat3[:,[0,4,6,1]] #['RMSE','NLL','CRPS','MAE']
Rank,All = [], []

for j in range(0,dat4.size(1)):
    tensor = dat4[:,j]
    namm = Metrics[j]
    data0 = []
    for group in range(tensor.shape[0]):
        for point in range(tensor.shape[1]):
            value, low, high = tensor[group, point]
            data0.append({
                #'Group': f'Group {group + 1}',
                'Group': new_list3[group],
                'Point': clean_names[point].capitalize(),
                'Value': value.item(),
                'Lower': low.item(),
                'Upper': high.item()
            })
    #
    df = pd.DataFrame(data0)
    df['Group'] = pd.Categorical(df['Group'], categories=df['Group'].unique(), ordered=True)
    df['ColorGroup'] = df['Group'].apply(assign_color)
    df_sorted = df.sort_values(by=['ColorGroup','Group'])
    All.append(df_sorted)
    p = (
        ggplot(df_sorted, aes(x='Group', y='Value',color='ColorGroup'))
        + geom_point()
        + geom_errorbar(aes(ymin='Lower', ymax='Upper'), width=0.3)
        #+ geom_text(aes(label=''), ha='left', nudge_y=0.1)
        + facet_wrap('~Point', nrow=1, scales='free_x')
        + coord_flip()
        + labs(
            y=namm,x='',
            subtitle=" "
        )
        + theme_minimal()
        + theme(
            panel_background=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            strip_background=element_blank(),
            panel_border=element_rect(color="black", fill=None, size=1),
            strip_text=element_text(weight='bold'),
            axis_text_y=element_text(margin={'r': 10},weight='bold',size=12),  # add space to right of y-axis labels
            axis_text_x =element_text(angle = 45, hjust = 1),
            legend_position='bottom',
            legend_direction='horizontal',
            legend_title=element_blank(),
            #axis_text_x=element_text(margin={'t': 10},weight='bold',size=20),
            #axis_title_y=element_text(weight='bold', size=24)
        )
    )
    Table = print(make_table(df_sorted))
    Rank.append(df_sorted[['Group','Value']].groupby('Group').mean())
    p.save(os.path.join(path0,'.'.join([namm,'png'])),bbox_inches='tight', dpi=500)
    #p.save(os.path.join(path0,'.'.join([namm,'png'])),width=10, height=4, dpi=300)
    #with open(os.path.join(path0,'.'.join([namm,'txt'])) as f:
        #f.write(Table)
    
