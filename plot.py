#plot

import gpytorch
import torch
import os
import math
import tqdm
import urllib.request
from scipy.io import loadmat
from math import floor
import pandas as pd
import numpy as np
import time
from pathlib import Path
from scipy.cluster.vq import kmeans2
from torch.nn import Linear
import torch.nn.functional as F
from gpytorch.means import ConstantMean, LinearMean
#from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution 
#from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
import gpytorch.settings as settings
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO, PredictiveLogLikelihood, ExactMarginalLogLikelihood,DeepPredictiveLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError, RelativeSquaredError
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gpytorch.models import ApproximateGP
import openpyxl
from uci_datasets import Dataset
from plotnine import *
#from uncertainties import ufloat
#import shap
#import inspect
#import gc
#Q = 8; num_tasks = 1; input_dims = 1; num_hidden_dgp_dims = 3; num_inducing = 256
from startup2 import MaskedMultitaskGaussianLikelihood, DGPHiddenLayer_CHOL, DGPHiddenLayer_DECOUPLED, DGPHiddenLayer_MEANFLD, DSPPHiddenLayer_MEANFLD, DSPPHiddenLayer_CHOL, DSPPHiddenLayer_DECOUPLED, MultitaskDSPP, MultitaskDeepGP, SVGPModel, ODSVGPModel, create_directory, data_load

path0 = os.getcwd() #"/rds/projects/o/oyebamok-deepmind-work/DGP_folder/batch_result7"
os.chdir(path0)
torch.manual_seed(2001210)
device = 'cuda'


#---------------------------------------plot
Metrics = ['RMSE','MAE','COVE','MSLL','NLL','PERC']
old_list = [f.name for f in Path(path0).iterdir() if f.is_file() and f.name.startswith('') and f.name.endswith('.txt')]
new_list = ['DSPP-Chol','DSPP-OD','DSPP-MF','DGP-Chol*','DGP-OD*','DGP-MF*','PPGPR-OD','PPGPR-Chol','DGP-Chol','DGP-OD','DGP-MF','ODSVGP','SVGP']
#names_no_ext = [os.path.splitext(f)[0] for f in files]
#['DeepPredictiveLogLikelihood_DSPPHiddenLayer_CHOL_DSPP', 'DeepPredictiveLogLikelihood_DSPPHiddenLayer_DECOUPLED_DSPP', 'DeepPredictiveLogLikelihood_DSPPHiddenLayer_MEANFLD_DSPP', 'PredictiveLogLikelihood_DGPHiddenLayer_CHOL_DGP', 'PredictiveLogLikelihood_DGPHiddenLayer_DECOUPLED_DGP', 'PredictiveLogLikelihood_DGPHiddenLayer_MEANFLD_DGP', 'PredictiveLogLikelihood_ODSVGPModel_GP', 'PredictiveLogLikelihood_SVGPModel_GP', 'VariationalELBO_DGPHiddenLayer_CHOL_DGP', 'VariationalELBO_DGPHiddenLayer_DECOUPLED_DGP', 'VariationalELBO_DGPHiddenLayer_MEANFLD_DGP', 'VariationalELBO_ODSVGPModel_GP', 'VariationalELBO_SVGPModel_GP']
files = [f.name for f in Path(path0).iterdir() if f.is_file() and f.name.startswith('combine') and f.name.endswith('.csv')]
clean_names = [os.path.splitext(f)[0].replace("combine_", "", 1) for f in files]
#clean_names = [os.path.splitext(f[17:])[0] for f in files]
datas = [torch.from_numpy(pd.read_csv(os.path.join(path0,f)).to_numpy()) for f in files]
dat = torch.stack(datas,2).reshape(2,datas[0].shape[0]//2,datas[1].shape[1],len(datas))
dat2 = torch.stack([dat[0], dat[0] - dat[1], dat[0] + dat[1]],3)
#

def assign_color(group):
    if "DSPP" in group:
        return "DSPP"
    elif "SVGP" in group or "PPGPR" in group:
        return "Single layer"
    elif "*" in group:
        return "This work"
    else:
        return "DGP"

for j in range(0,dat2.size(1)):
    tensor = dat2[:,j]
    namm = Metrics[j]
    data0 = []
    for group in range(tensor.shape[0]):
        for point in range(tensor.shape[1]):
            value, low, high = tensor[group, point]
            data0.append({
                #'Group': f'Group {group + 1}',
                'Group': new_list[group],
                'Point': clean_names[point].capitalize(),
                'Value': value.item(),
                'Lower': low.item(),
                'Upper': high.item()
            })
    #
    df = pd.DataFrame(data0)
    df['Group'] = pd.Categorical(df['Group'], categories=df['Group'].unique(), ordered=True)
    df['ColorGroup'] = df['Group'].apply(assign_color)
    
    p = (
        ggplot(df, aes(x='Group', y='Value',color='ColorGroup'))
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
            axis_text_y=element_text(margin={'r': 10},weight='bold',size=10),  # add space to right of y-axis labels
            legend_position='bottom',
            legend_direction='horizontal',
            legend_title=element_blank(),
            #axis_text_x=element_text(margin={'t': 10},weight='bold',size=20),
            #axis_title_y=element_text(weight='bold', size=24)
        )
    )
    
    #print(p)
    p.save(os.path.join(path0,'.'.join([namm,'png'])),width=10, height=4, dpi=300)

#--------------------------END
p = (
    ggplot(df, aes(x='Group', y='Value'))
    + geom_point()
    + geom_errorbar(aes(ymin='Lower', ymax='Upper'), width=0.3)
    + geom_text(aes(label='Label'), ha='left', nudge_y=0.1)
    + facet_wrap('~Point', nrow=1, scales='free_y')
    + coord_flip()
    + labs(
        y=namm,
        subtitle="Error Bars by Group and Point"
    )
    + theme_minimal()
    + theme(
        panel_background=element_blank(),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        panel_border=element_rect(color="black", fill=None, size=1),
        strip_text=element_text(weight='bold'),
        axis_text_y=element_text(margin={'r': 15})  # increased right margin for more space
    )
)

#-------------------------------------------------------------
# Example tensor [group, point, (value, lower, upper)]
tensor = torch.randn(6, 13, 3) * 2 + 10  # shape: [6, 13, 3]
# Convert to DataFrame
data0 = []
for group in range(tensor.shape[0]):
    for point in range(tensor.shape[1]):
        value, low, high = tensor[group, point]
        data0.append({
            'Group': f'Group {group + 1}',
            'Point': point + 1,
            'Value': value.item(),
            'Lower': low.item(),
            'Upper': high.item()
        })

df = pd.DataFrame(data0)

# Ensure 'Point' is a categorical type and sorted the way you want
df['Point'] = pd.Categorical(df['Point'], categories=sorted(df['Point'].unique()), ordered=True)

p = (
    ggplot(df, aes(x='Group', y='Value'))
    + geom_point()
    + geom_errorbar(aes(ymin='Lower', ymax='Upper'), width=0.3)
    + facet_wrap('~Point', nrow=1, scales='free_y')
    + coord_flip()
    + labs(
        y=namm,
        subtitle="Error Bars by Group and Point"
    )
    + theme_minimal()
    + theme(
        panel_background=element_blank(),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        panel_border=element_rect(color="black", fill=None, size=1),
        strip_text=element_text(weight='bold'),
        axis_text_y=element_text(margin={'r': 10})  # add space to right of y-axis labels
    )
)

print(p)


import pandas as pd

# Define the desired order
desired_order = ['Group A', 'Group B', 'Group C']  # Replace with your actual order

# Convert Group column to ordered category
df['Group'] = pd.Categorical(df['Group'], categories=desired_order, ordered=True)



import seaborn as sns

def plot_gp_2d(x, mean, stddev, title="GP Mean + Uncertainty"):
    x1 = x[:, 0].detach().numpy()
    x2 = x[:, 1].detach().numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=x1, y=x2, hue=mean.detach().numpy(), palette='viridis')
    plt.title("Predicted Mean")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=x1, y=x2, hue=stddev.detach().numpy(), palette='coolwarm')
    plt.title("Predicted StdDev (Uncertainty)")
    plt.suptitle(title)
    plt.show()






import matplotlib.pyplot as plt
import seaborn as sns

import torch

def crps_mc(samples, y_true):
    """
    Monte Carlo CRPS approximation for multitask Gaussian processes.
    samples: Tensor of shape (n_samples, batch_size, num_tasks)
    y_true:  Tensor of shape (batch_size, num_tasks)
    """
    n_samples = samples.size(0)
    abs_diff = torch.abs(samples - y_true.unsqueeze(0))  # (n_samples, batch, task)
    term1 = abs_diff.mean(dim=0)
    pairwise_diffs = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1))  # (n, n, batch, task)
    term2 = pairwise_diffs.mean(dim=(0, 1)) / 2.0
    crps = term1 - term2  # (batch_size, num_tasks)
    return crps.mean()  # mean across all tasks and samples

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(100):
    preds = likelihood(model(test_x))  # MultivariateNormal
    samples = preds.rsample(torch.Size([100]))  # (100, batch_size, num_tasks)
    samples2 = preds.rsample() 
    crps_score = crps_mc(samples, test_y)
    print(f"Mean CRPS (Multi-task): {crps_score.item():.4f}")


def plot_crps_per_task(samples, y_true, task_names=None):
    """
    Visualize CRPS per task
    """
    crps_per_task = crps_mc(samples, y_true)  # (batch_size, num_tasks)
    crps_mean_per_task = crps_per_task.mean(dim=0)

    plt.figure(figsize=(10, 3))
    sns.barplot(x=list(range(crps_mean_per_task.size(0))),
                y=crps_mean_per_task.cpu().numpy())
    plt.ylabel("CRPS")
    plt.xlabel("Task")
    if task_names:
        plt.xticks(ticks=range(len(task_names)), labels=task_names)
    plt.title("CRPS per Task")
    plt.tight_layout()
    plt.show()



#------------------------------------
library(ggplot2)
library(tidyr)

# Example data (creating a tensor-like structure in R)
set.seed(123)
tensor_data <- data.frame(
  X = rnorm(100),  # x-coordinates
  Y = rnorm(100),  # y-coordinates
  Z = rnorm(100)   # z-coordinates (error bars or another axis)
)

# Plotting with ggplot2
ggplot(tensor_data, aes(x = X, y = Y)) +
  geom_point(color = "blue") +  # Scatter plot
  geom_errorbar(aes(ymin = Y - Z, ymax = Y + Z), width = 0.1, color = "red") +  # Error bars
  theme_minimal() +
  labs(title = "Scatter Plot with Error Bars", x = "X", y = "Y")







#combine results
#Metrics.to_csv(os.path.join(path1,'.'.join(['_'.join(["Metric",nam,obj.__name__,layer.__name__,Type]),'txt'])), index=False)

for j in range(0,len(uci)):
    nam = uci[j]
    path1 = os.path.join(path0,'_'.join(['result', nam]))
    #namm = os.path.join(path1,'.'.join(['_'.join(["Metric",nam,obj.__name__,layer.__name__,Type]),'txt']))    
    for Type in Types:
        for obj in obj_fun:
            if Type== 'GP':
                res = []
                for layer in layer_type3:
                    namm = os.path.join(path1,'.'.join(['_'.join(["Metric",nam,obj.__name__,layer.__name__,Type]),'txt'])) 
                    dat = pd.read_csv(namm)
                    res.append(dat)
                    return res
            elif Type== 'DGP':
                for layer in layer_type1:
                    namm = os.path.join(path1,'.'.join(['_'.join(["Metric",nam,obj.__name__,layer.__name__,Type]),'txt']))
        if Type== 'DSPP':
            for layer in layer_type2:
                namm = os.path.join(path1,'.'.join(['_'.join(["Metric",nam,obj.__name__,layer.__name__,Type]),'txt']))
