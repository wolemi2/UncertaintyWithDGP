#pip install gpytorch==1.8 linear_operator==0.3
#!pip install urllib datetime openpyxl properscoring plotnine scipy pandas numpy
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
import datetime
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
import properscoring as ps
import linear_operator
from uci_datasets import Dataset
#from plotnine import *
#from uncertainties import ufloat
#import shap
#import inspect
#import gc
from Utils import MaskedMultitaskGaussianLikelihood, DGPHiddenLayer_CHOL, DGPHiddenLayer_DECOUPLED, DGPHiddenLayer_MEANFLD, DSPPHiddenLayer_MEANFLD, DSPPHiddenLayer_CHOL, DSPPHiddenLayer_DECOUPLED, MultitaskDSPP, MultitaskDeepGP, SVGPModel, ODSVGPModel, create_directory, data_load, img_to_patch, mask_patches, date_phase

path0 = os.getcwd() #
os.chdir(path0)
torch.manual_seed(2001210)
device = 'cuda'
scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0.,1.)
#------------------------------fixed parameters
eps = 0.000001
TEST_SPLIT = 0.80
split = int(TEST_SPLIT*10)
uci = ['elevators','bike','protein','energy','Jura','3droad','IAP','song']#
BATCH = [1024,1024,1024,128,128,1024*2,1024*2,1024*2]
num_hidden_dgp_dims = hidden_dim = 3
num_epochs = 400
initial_lr = lr = 0.01
num_samples = 50
num_inducing = 128
Q = num_quadrature_sites = 8
milestones = [50, 100, 150]
ee = 1e-1
skip = 20
nlpd = gpytorch.metrics.negative_log_predictive_density
msll = gpytorch.metrics.mean_standardized_log_loss
cove = gpytorch.metrics.quantile_coverage_error


layer_type1 = [DGPHiddenLayer_CHOL, DGPHiddenLayer_MEANFLD]
layer_type2 = [DSPPHiddenLayer_CHOL,DSPPHiddenLayer_MEANFLD]
layer_type3 = [ODSVGPModel,SVGPModel]
obj_fun = [PredictiveLogLikelihood,VariationalELBO,DeepPredictiveLogLikelihood]
Types = ['DGP','GP','DSPP']
#------------------------------------------

def train(epoch,train_loader,optimizer,mll,mask,Q):
    model.train()
    model.likelihood.train()
    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(num_samples),gpytorch.settings.cholesky_jitter(float_value=ee, double_value=ee, half_value=None),gpytorch.settings.cg_tolerance(5),gpytorch.settings.cholesky_max_tries(5000), gpytorch.settings.max_cg_iterations(5000):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            with linear_operator.settings.max_cg_iterations(5000), linear_operator.settings.cg_tolerance(5.0), linear_operator.settings.cholesky_jitter(1e-3):
                optimizer.zero_grad()
                output = model(data.clone().to(torch.float32))
                #output = model(data.to(torch.float))
                #loss = -mll(output, target).sum()
                if mask is not None:
                    loss = -mll(output, target,mask=mask.to(device),Q=Q,Type=Type).sum()
                else:
                    loss = -mll(output, target).sum()
                loss.backward()
                optimizer.step()
                minibatch_iter.set_postfix(loss=loss.item())    


def mod(model,mask):
    if torch.cuda.is_available():
        model = model.to(device)
    #define loss function, and evaluation metrics------------------------
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=initial_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    #
    Met = []
    start = time.time()
    for epoch in range(0, num_epochs+1): 
        train(epoch,train_loader,optimizer,mll,mask,Q)
        outt = test(test_loader)
        scheduler.step()
        Met.append(outt[0])
    #
    duration = time.time() - start
    time0 = duration/60
    #print((time0))
    #print(Met)
    R = outt[1]
    Metrics = pd.DataFrame(np.stack(Met,0))
    Metrics.columns = ['rmse','mae','cove','msll','nlpd','perc','crps']
    torch.save(model.state_dict(), os.path.join(path1,'.'.join(['_'.join(["Model",nam,obj.__name__,layer.__name__,Type]),'pth'])))  
    Metrics.to_csv(os.path.join(path1,'.'.join(['_'.join(["Metric",nam,obj.__name__,layer.__name__,Type]),'txt'])), index=False)
    np.savetxt(os.path.join(path1,'.'.join(['_'.join(["Time0",nam,obj.__name__,layer.__name__,Type]),'txt'])),np.array([time0]))
    if nam=='IAP':
        with pd.ExcelWriter(os.path.join(path1,'.'.join(['_'.join(["All_Result",nam,obj.__name__,layer.__name__,Type]),'xlsx']))) as writer:
                pd.DataFrame(R[0].cpu()).to_excel(writer, sheet_name='Mean')
                pd.DataFrame(R[1].cpu()).to_excel(writer, sheet_name='VAR')
                pd.DataFrame(R[2].cpu()).to_excel(writer, sheet_name='YY') 
    elif nam=='ndvi':
        Mean = pd.DataFrame(R[0].cpu())
        Mean.to_csv(os.path.join(path1,'.'.join(['_'.join(["All_result_Mean",nam,obj.__name__,layer.__name__,Type]),'csv'])), index=False)
        VAR = pd.DataFrame(R[1].cpu())
        VAR.to_csv(os.path.join(path1,'.'.join(['_'.join(["All_result_VAR",nam,obj.__name__,layer.__name__,Type]),'csv'])), index=False)
        YY = pd.DataFrame(R[2].cpu())
        YY.to_csv(os.path.join(path1,'.'.join(['_'.join(["All_result_YY",nam,obj.__name__,layer.__name__,Type]),'csv'])), index=False)
        torch.save(train_loader.dataset,os.path.join(path1,'.'.join(['_'.join(["dataset",nam,obj.__name__,layer.__name__,Type]),'pth'])))
        #dataset = torch.load('dataset.pth')


def ensure_2d(tensor):  
    if tensor.dim() == 0:
        return tensor.view(1, 1)
    elif tensor.dim() == 1:
        return tensor.view(1, -1)
    return tensor


def test(test_loader):
    R1, R2, R3 = [], [], []
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_samples),gpytorch.settings.cholesky_jitter(float_value=ee, double_value=ee, half_value=None):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
                #data, target = next(iter(test_loader))
            #
            if Type== 'DSPP':
                f_dist = model(data.clone().to(torch.float32))
                means, vars, ll, Msll, Cove = model.predict(data.clone().to(torch.float32),target)
                weights = model.quad_weights.unsqueeze(-1).unsqueeze(1).exp()#.cpu()
                rmse = ((weights * means).sum(0) - target.to(torch.float)).pow(2.0).mean().sqrt()
                mae = torch.mean(torch.abs((weights * means).sum(0) - target.to(torch.float)))
                #
                ll = -ll.mean().item()
                perc = (f_dist.variance/vars).mean()
                #crps = ps.crps_gaussian(target.to(torch.float).numpy(),mu=(weights * means).sum(0).numpy(),sig=(weights * vars).sum(0).sqrt().numpy())
                crps = ps.crps_gaussian(target.to(torch.float).cpu(),mu=(weights * means).sum(0).cpu(),sig=(weights * vars).sum(0).sqrt().cpu())
                met = np.array([rmse.item(),mae.item(),Cove.mean().item(),Msll.mean().item(),ll,perc.item(),crps.mean()])
                #met = np.array([rmse.item(),mae.item(),Cove.mean().item(),Msll.mean().item(),Nlpd.mean().item(),ll])
                R1.append((weights * means).sum(0))
                R2.append((weights * vars).sum(0))
                #ensure_2d(target).t()
                R3.append(ensure_2d(target))               
                met = np.round(met, 4)
            else:    
                f_dist = model(data.clone().to(torch.float32))
                output = model.likelihood(f_dist)  # This gives us num_samples samples from the predictive distribution
                orig = target#.flatten()
                mae = torch.mean(torch.abs(output.mean - orig))
                rmse = torch.mean(torch.pow(output.mean - orig, 2)).sqrt()
                Nlpd = model.likelihood.log_marginal(orig.to(torch.float), f_dist).mean().detach()
                #Nlpd = nlpd(output,orig.to(torch.float)).mean().detach()
                #model.to(torch.float)
                #-pred_dist.log_prob(test_y) / test_y.shape[combine_dim]
                Msll = msll(output,orig).mean().detach()
                Cove = cove(output,orig).mean().detach()
                perc = (f_dist.variance/output.variance).mean() 
                crps = ps.crps_gaussian(x=orig.cpu(),mu=output.mean.cpu(),sig=output.variance.sqrt().cpu()).mean()
                #crps = ps.crps_gaussian(x=orig.detach().numpy(),mu=output.mean.detach().numpy(),sig=output.variance.sqrt().detach().numpy()).mean()
                met = np.array([rmse.cpu().item(),mae.cpu().item(),Cove.cpu().item(),Msll.cpu().item(),-Nlpd.cpu().item(),perc.cpu(),crps.item()])
                R1.append(output.mean)
                R2.append(output.variance)
                R3.append(ensure_2d(orig).t())  
                met = np.round(met, 4)
        #R = (torch.vsack(R1),torch.vstack(R2),torch.vstack(R3))
        if Type=='DSPP':
            R = (torch.vstack(R1),torch.vstack(R2),torch.vstack(R3))
        elif Type=='DGP':
            R = (torch.concat(R1,1).mean(0),torch.concat(R2,1).mean(0),torch.concat(R3,1))
        else:
            R = (torch.concat(R1,0),torch.concat(R2,0),torch.concat(R3,1))

    return met, R

    
def combined_metrics(path):
    path1 = path
    file_names = os.listdir(path1)
    #files_only = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]
    #txt_files = [f for f in os.listdir(path1) if f.endswith('.txt') and os.path.isfile(os.path.join(path1, f))]
    files = [f.name for f in Path(path1).iterdir() if f.is_file() and f.name.startswith('Met') and f.name.endswith('.txt')]
    print(sorted(files))
    dataframes = [torch.from_numpy(pd.read_csv(os.path.join(path1,f)).to_numpy()) for f in sorted(files)]
    dat = torch.stack(dataframes,2)
    if num_epochs > skip:
        dat = dat[skip:,:]    
    mett = pd.DataFrame(torch.concat([dat.mean(0).t(),dat.std(0).t()]))
    mett.columns = ['rmse','mae','cove','msll','nlpd','perc','crps']
    mett.to_csv(os.path.join(path0,'.'.join(['_'.join(['combine',nam]),'csv'])), index=False)
    clean_names = [os.path.splitext(f[17:])[0] for f in sorted(files)]
    with open(os.path.join(path0,'.'.join([nam,'txt'])), "w") as f:
        for name in clean_names:
            f.write(name + "\n")


for j in range(0,len(uci)):
    nam = uci[j]
    path1 = os.path.join(path0,'_'.join(['result', nam]))
    create_directory(path1)
    batch_size = BATCH[j]
    train_loader, test_loader,inducing_points, mask = data_load(uci[j],batch_size,split)
    num_tasks = len(train_loader.dataset[0][1])
    input_dims = len(train_loader.dataset[0][0])
    output_dims = num_tasks
    train_x_shape = len(train_loader.dataset),len(train_loader.dataset[0][0])
    print(input_dims)
    print(num_tasks)
    for Type in Types:
        for obj in obj_fun[0:2]:
            if Type== 'GP':
                for layer in layer_type3:                                        
                    model = layer(num_tasks,num_inducing, input_dims,mask=mask)                    
                    mll = obj(model.likelihood, model, num_data=train_x_shape[0])
                    mod(model,mask)
            elif Type== 'DGP':
                for layer in layer_type1:
                    model = MultitaskDeepGP(train_x_shape,layer,output_dims,hidden_dim,num_inducing,mask=mask)
                    mll = DeepApproximateMLL(obj(model.likelihood, model, num_data=train_x_shape[0])) 
                    mod(model,mask)
        if Type== 'DSPP':
            obj = obj_fun[2]
            for layer in layer_type2:
                model = MultitaskDSPP(layer, train_x_shape, output_dims=num_tasks, inducing_points=inducing_points, num_inducing=num_inducing, hidden_dim=hidden_dim, mask=mask,Q=Q).to(torch.float)
                mll = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=train_x_shape[0])
                mod(model,mask)
                #
    #combined_metrics(path1)





    

