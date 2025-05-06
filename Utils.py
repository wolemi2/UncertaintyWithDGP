#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####------------------------GP module 
import os
import gpytorch
import torch
import math
from torch import nn
import pandas as pd
import numpy as np
from torch import masked_fill
import datetime
from scipy.cluster.vq import kmeans2
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
import gpytorch.settings as settings
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from uci_datasets import Dataset
import gc
scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0.,1.)
nlpd = gpytorch.metrics.negative_log_predictive_density
msll = gpytorch.metrics.mean_standardized_log_loss
cove = gpytorch.metrics.quantile_coverage_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path0 = os.getcwd()
num_inducing = 256
#------------------------------------------------------------------------------------------

def data_load(uu,batch_size,split):
    if uu=='Jura':
        dir0 = os.path.join(path0,"Jura.csv")
        out0 = pd.read_csv(dir0,index_col=0)
        Y = out0.loc[:, ['Cd', 'Co', 'Cu']] 
        X = out0.loc[:,out0.columns[:14]]
        Y = (Y - Y.mean(0)) / (Y.std(0))
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        Y = torch.tensor(Y.to_numpy()).clone().to(torch.float32)
        X = torch.tensor(X.to_numpy()).clone().to(torch.float32)
        ind = np.arange(0,Y.size(0))
        train_n = int(math.floor(0.1 * split * Y.size(0)))
        index = np.random.choice(ind, train_n,replace=False)
        train_x = X[index, :].contiguous()
        train_y = Y[index,:].contiguous()
        test_x = np.delete(X,index, axis=0).contiguous()
        test_y = np.delete(Y,index, axis=0).contiguous()
        inducing_points = (train_x[torch.randperm(min(1000 * 100, train_y.size(0)))[0:num_inducing], :])
        inducing_points = inducing_points.clone().data.cpu().numpy()
        inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),inducing_points, minit='matrix')[0]).to(device).to(torch.float)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        mask = None
    elif uu=='IAP':
        dir0 = os.path.join(path0,"Datanew2.xlsx")
        out0 = pd.read_excel(dir0,skiprows=0,sheet_name=None)# 4 scenario for 2050s#[9,10,11])
        id = [2, 3, 4, 5, 6, 7,9,10, 11, 12, 13, 14, 15, 16,17]
        train_y = torch.tensor(out0['train_y'].values)
        train_x = torch.tensor(out0['train_x'].values)
        test_y = torch.tensor(out0['test_y'].values)
        test_x = torch.tensor(out0['test_x'].values)
        nam0 = out0['train_y'].columns
        nam1 = out0['train_x'].columns
        inducing_points = (train_x[torch.randperm(min(1000 * 100, train_y.size(0)))[0:num_inducing], :])
        inducing_points = inducing_points.clone().data.cpu().numpy()
        inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),inducing_points, minit='matrix')[0]).to(torch.float)
        #inducing_indices = torch.randperm(train_x.size(0))[:num_inducing]
        #inducing_points = train_x[inducing_indices].to(device).to(torch.float32)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        mask = None
    elif uu=='ndvi':
        num_channels = 3
        data = pd.read_csv(os.path.join(path0, 'Bartley.csv'),skiprows=0)
        ind_x, ind_y, ind_t = len(np.unique(data['x'])), len(np.unique(data['y'])), len(np.unique(data['Date']))
        data = data.to_numpy()
        data0 = np.asarray(data[:,[1,2,3,4,5]]).astype(np.float32)    
        #
        t0 = [date_phase(date) for date in data[:,0]]
        t1 = np.sin(2 * np.pi * pd.to_datetime(data[:,0]).month / 12)
        t2 = np.cos(2 * np.pi * pd.to_datetime(data[:,0]).month / 12)
        inp = np.column_stack((t0,t1,t2,data0))    
        data0 = torch.from_numpy(inp)
        data1 = data0.view(ind_t,ind_x,ind_y,data0.shape[1])
        Y0 = scale_to_bounds(data1).permute(0,3,1,2)
        train_x = Y0[:,:3].permute(0,3,2,1).reshape(-1,3)
        test_y = Y0[:,5:].clone().type(torch.int32).permute(0,3,2,1).reshape(-1,num_channels)
        #reshape(446,64,64,3)
        Img = [mask_patches(Y0[:,5+k],mask_ratio=.25)[0] for k in [0,1,2]]
        train_y = torch.stack(Img,axis=3).reshape(-1,num_channels)
        mask = torch.isnan(train_y).reshape(-1,num_channels)
        inducing_points = (train_x[torch.randperm(min(1000 * 100, train_y.size(0)))[0:num_inducing], :])
        inducing_points = inducing_points.clone().data.cpu().numpy()
        inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),inducing_points, minit='matrix')[0]).to(device).to(torch.float)
        train_loader = torch.utils.data.DataLoader(TensorDataset(train_x,train_y), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(TensorDataset(train_x,test_y), batch_size=batch_size, shuffle=False)
        del data0, data1, data, t0,t1,t2,inp, Y0,Img,train_x,train_y,test_y
        gc.collect()
    else:          
        data = Dataset(uu)
        x_1, y_1, x_2, y_2 = data.get_split(split)
        #x_1, y_1, x_2, y_2 = torch.tensor(x_1), torch.tensor(y_1),torch.tensor(x_2), torch.tensor(y_2)
        y_1, y_2 = torch.tensor(y_1), torch.tensor(y_2)
        train_x = (x_1 - x_1.min(0)) / (x_1.max(0) - x_1.min(0)) #x_1.sub(x_1.mean(0)).div(x_1.std(0)+eps)
        test_x = (x_2 - x_1.min(0)) / (x_1.max(0) - x_1.min(0))
        train_y = (y_1 - y_1.mean(0))/y_1.std(0)
        test_y = (y_2 - y_1.mean(0))/y_1.std(0)
        train_x,test_x = torch.tensor(train_x),torch.tensor(test_x)
        inducing_points = (train_x[torch.randperm(min(1000 * 100, train_y.size(0)))[0:num_inducing], :])
        inducing_points = inducing_points.clone().data.cpu().numpy()
        inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),inducing_points, minit='matrix')[0]).to(device).to(torch.float)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        mask = None
    return train_loader, test_loader, inducing_points, mask


def create_directory(directory_path):
  if not os.path.exists(directory_path):
    try:
      os.makedirs(directory_path)
      print(f"Directory created successfully at: {directory_path}")
    except OSError as e:
      print(f"Error creating directory at {directory_path}: {e}")
  else:
    print(f"Directory already exists at: {directory_path}")

def date_phase(input_date):
    date0 = datetime.datetime.strptime(input_date, "%Y-%m-%d")
    year_start = datetime.datetime(year=date0.year, month=1, day=1)
    year_end = datetime.datetime(year=date0.year, month=12, day=31)
    phase = (date0 - year_start).days / (year_end - year_start).days
    return phase


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x
    

def mask_patches(images, mask_ratio=0.5):
    B, P, D = images.shape
    mask = torch.rand(B, P) < mask_ratio  # Boolean mask (True = masked)
    masked_images = images.clone()
    masked_images[mask] = float('nan') #0  # Zero out masked patches
    return masked_images, mask

class DGPHiddenLayer_CHOL(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])
        #
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        #
        
        #self.mean_module = gpytorch.means.MultitaskMean(ConstantMean(),num_tasks) if linear_mean else gpytorch.means.MultitaskMean(LinearMean(input_dims),num_tasks)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DGPHiddenLayer_MEANFLD(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])
        #
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        #
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
        #
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DGPHiddenLayer_MAP(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])
        #
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        #
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            num_tasks, rank=1
        )
        #
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def make_orthogonal_old(self, input_dims, output_dims, num_inducing, batch_shape):
    mean_inducing_points = torch.randn(output_dims,int(3*num_inducing), input_dims)
    covar_inducing_points = torch.randn(output_dims,num_inducing, input_dims)
    if torch.cuda.is_available():
        mean_inducing_points = mean_inducing_points.cuda()
        covar_inducing_points = covar_inducing_points.cuda()
    #
    covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        self, covar_inducing_points,
        gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2),batch_shape=batch_shape),
        learn_inducing_locations=True
    )
    #
    variational_distribution = gpytorch.variational.DeltaVariationalDistribution(num_inducing_points=mean_inducing_points.size(-2),batch_shape=batch_shape)
    variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        covar_variational_strategy, mean_inducing_points, variational_distribution    
    )
    return variational_strategy


class DGPHiddenLayer_DECOUPLED(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, linear_mean=True):
        batch_shape = torch.Size([output_dims])
        variational_strategy = make_orthogonal(self, input_dims, output_dims,num_inducing, batch_shape)
        super().__init__(variational_strategy, input_dims, output_dims)
        #
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
        #
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape,layer,output_dims,hidden_dim,num_inducing,mask):
        n = train_x_shape[0]
        self.n = n
        #
        hidden_layer = layer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,num_inducing=num_inducing,
            linear_mean=True
        )
        last_layer = layer(
            input_dims=hidden_layer.output_dims,
            output_dims=output_dims,num_inducing=num_inducing,
            linear_mean=False
        )
        #
        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        # multitask likelihood instead of the standard GaussianLikelihood
        #self.likelihood = MultitaskGaussianLikelihood(num_tasks=output_dims,rank=1,has_global_noise=False,has_task_noise=True)  
        if mask is not None:
            self.likelihood = MaskedMultitaskGaussianLikelihood(num_tasks=output_dims,rank=1,has_global_noise=False,has_task_noise=True)
        else:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=output_dims,rank=1,has_global_noise=False,has_task_noise=True)  
    #
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

class ODSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self,num_tasks,num_inducing, input_dims,mask):
        batch_shape = torch.Size([num_tasks])
        mean_inducing_points = torch.randn(num_tasks,int(3*num_inducing), input_dims)
        covar_inducing_points = torch.randn(num_tasks,num_inducing, input_dims)
        #
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2),batch_shape=torch.Size([num_tasks]))
        variational_distribution2 = gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2),batch_shape=torch.Size([num_tasks]))
        #
        covar_variational_strategy = gpytorch.variational.VariationalStrategy(self, covar_inducing_points, variational_distribution,learn_inducing_locations=True)
        variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(covar_variational_strategy, mean_inducing_points,variational_distribution2)
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strategy,num_tasks,task_dim=-1)
        super().__init__(variational_strategy)        
        self.mean_module = gpytorch.means.LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
        #self.likelihood = MultitaskGaussianLikelihood(num_tasks,rank=1,has_global_noise=False,has_task_noise=True)
        if mask is not None:
            self.likelihood = MaskedMultitaskGaussianLikelihood(num_tasks,rank=1,has_global_noise=False,has_task_noise=True)
        else:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks,rank=1,has_global_noise=False,has_task_noise=True)  
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self,num_tasks,num_inducing, input_dims,mask):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_tasks,num_inducing, input_dims)
        batch_shape = torch.Size([num_tasks])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )
        
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks,task_dim=-1
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
        #self.likelihood = MultitaskGaussianLikelihood(num_tasks,rank=1,has_global_noise=False,has_task_noise=True)
        if mask is not None:
            self.likelihood = MaskedMultitaskGaussianLikelihood(num_tasks,rank=1,has_global_noise=False,has_task_noise=True)
        else:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks,rank=1,has_global_noise=False,has_task_noise=True)  
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MaskedMultitaskGaussianLikelihood(gpytorch.likelihoods.MultitaskGaussianLikelihood):
    def expected_log_prob(self, target, input, mask, Type,*params, **kwargs):
        if Type == 'DGP':
            mask = torch.isnan(target)
            target = target.masked_fill(target.isnan(), -999.)
            mean, variance = input.mean, input.variance
            mask = torch.stack([mask] * len(mean))
            num_event_dim = len(input.event_shape)
            #
            noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
            # Potentially reshape the noise to deal with the multitask case
            noise = noise.view(*noise.shape[:-1], *input.event_shape).to(device)
            #
            res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
            res = res.mul(-0.5)
            #res = res.nan_to_num().mul(~mask) #this is also correct
            res = res[~mask]
            if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
                res = res.sum(list(range(-1, -num_event_dim, -1)))
            return res
        elif Type =='GP':
            #target = torch.stack([target] * Q)
            mask = torch.isnan(target)
            target = target.masked_fill(target.isnan(), -999.)  
            mean, variance = input.mean, input.variance
            num_event_dim = len(input.event_shape)
            #
            noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
            # Potentially reshape the noise to deal with the multitask case
            noise = noise.view(*noise.shape[:-1], *input.event_shape).to(device)
            res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
            res = res.mul(-0.5)
            #res = res.nan_to_num().mul(~mask) #this is also correct
            res = res[~mask]
            if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
                res = res.sum(list(range(-1, -num_event_dim, -1)))
            return res

    def log_marginal(self, target, input, mask, Q, Type='DSPP',*params, **kwargs):
        if Type=='GP':
        #if not isinstance(Type, DSPP):
            mask = torch.isnan(target)
            target = target.masked_fill(target.isnan(), -999.)   
            # Mean and covariance of the multitask GP output
            num_tasks = target.shape[-1]        
            # Mean and covariance of the multitask GP output
            mean_f = input.mean  # (batch_size, num_tasks)
            cov_f = input.variance 
            # Likelihood noise covariance
            noise_covar = self._shaped_noise_covar(mean_f.shape).diag()
            # Full covariance of observed data
            full_covar = cov_f.view(-1) + noise_covar  # (batch_size * num_tasks, batch_size * num_tasks)
            # Compute log likelihood term
            mvn = gpytorch.distributions.MultivariateNormal(mean_f.view(-1)[~mask.flatten()], torch.diag(full_covar[~mask.flatten()]))
            #log_marginal = mvn.log_prob(target.view(-1)[~mask.flatten()])  # Log marginal likelihood 
            marginal = target.view(-1)[~mask.view(-1)]
            log_marginal = mvn.log_prob(marginal.type(torch.int64))
            return log_marginal
        elif Type=='DGP':
            mask = torch.isnan(target)
            target = target.masked_fill(target.isnan(), -999.)  
            num_tasks = target.shape[-1]        
            # Mean and covariance of the multitask GP output
            mean_f = input.mean  # (batch_size, num_tasks)
            cov_f = input.variance 
            mask = torch.stack([mask] * len(mean_f))
            target = torch.stack([target] * len(mean_f))
            # Likelihood noise covariance
            noise_covar = self._shaped_noise_covar(mean_f.shape)#.diag()
            # Full covariance of observed data
            full_covar = cov_f.view(-1) #+ noise_covar.view(-1)  # (batch_size * num_tasks, batch_size * num_tasks)
            # Compute log likelihood term
            mvn = gpytorch.distributions.MultivariateNormal(mean_f.view(-1)[~mask.flatten()], torch.diag(full_covar[~mask.flatten()]))
            #log_marginal = mvn.log_prob(target.view(-1)[~mask.flatten()])  # Log marginal likelihood 
            marginal = target.view(-1)[~mask.view(-1)]
            log_marginal = mvn.log_prob(marginal.type(torch.int64))
            return log_marginal
        elif Type=='DSPP':
            target = torch.stack([target] * Q)
            mask = torch.isnan(target)
            target = target.masked_fill(target.isnan(), -999.)   
            # Mean and covariance of the multitask GP output
            num_tasks = target.shape[-1]        
            # Mean and covariance of the multitask GP output
            mean_f = input.mean  # (batch_size, num_tasks)
            cov_f = input.variance 
            # Likelihood noise covariance
            noise_covar = self._shaped_noise_covar(mean_f.shape)#.diag()
            # Full covariance of observed data
            full_covar = cov_f.view(-1) #+ noise_covar.view(-1)  # (batch_size * num_tasks, batch_size * num_tasks)
            # Compute log likelihood term
            mvn = gpytorch.distributions.MultivariateNormal(mean_f.view(-1)[~mask.flatten()], torch.diag(full_covar[~mask.flatten()]))
            #log_marginal = mvn.log_prob(target.view(-1)[~mask.flatten()])  # Log marginal likelihood   
            marginal = target.view(-1)[~mask.view(-1)]
            log_marginal = mvn.log_prob(marginal.type(torch.int64))
            return log_marginal

#if not isinstance(model, DSPP):
#------------------------------------------------------------------------------------------
class DSPPHiddenLayer_MEANFLD(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, Q,inducing_points=None, mean_type='constant'):
        batch_shape = torch.Size([output_dims])
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)
        #
        # Let's use mean field / diagonal covariance structure.
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape if output_dims is not None else torch.Size([])
        )
        #
        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        #
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        super().__init__(variational_strategy, input_dims, output_dims, Q)
        #
        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        #
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
    #
    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DSPPHiddenLayer_CHOL(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, Q,inducing_points=None, mean_type='constant'):
        batch_shape = torch.Size([output_dims])
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)
        #
        # 
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape if output_dims is not None else torch.Size([])
        )
        #
        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        #
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        super().__init__(variational_strategy, input_dims, output_dims, Q)
        #
        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        #
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
    #
    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DSPPHiddenLayer_MAP(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, Q,inducing_points=None, mean_type='constant'):
        batch_shape = torch.Size([output_dims])
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)
        #
        # 
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape if output_dims is not None else torch.Size([])
        )
        #
        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        #
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        #super(DSPPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims, Q)
        super().__init__(variational_strategy, input_dims, output_dims, Q)
        #
        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        #
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
    #
    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_orthogonal(self, input_dims, output_dims, num_inducing, batch_shape):
    mean_inducing_points = torch.randn(output_dims,int(3*num_inducing), input_dims)
    covar_inducing_points = torch.randn(output_dims,num_inducing, input_dims)
    if torch.cuda.is_available():
        mean_inducing_points = mean_inducing_points.to(device)
        covar_inducing_points = covar_inducing_points.to(device)
    #
    covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        self, covar_inducing_points,
        CholeskyVariationalDistribution(covar_inducing_points.size(-2),batch_shape=batch_shape if output_dims is not None else torch.Size([])),learn_inducing_locations=True
    )
    #
    variational_distribution = gpytorch.variational.DeltaVariationalDistribution(num_inducing_points=mean_inducing_points.size(-2),batch_shape=batch_shape if output_dims is not None else torch.Size([]))
    variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        covar_variational_strategy, mean_inducing_points, variational_distribution    
    )
    return variational_strategy


class DSPPHiddenLayer_DECOUPLED(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, Q,inducing_points=None, mean_type='constant'):
        batch_shape = torch.Size([output_dims])
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)
        #
        # 
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape if output_dims is not None else torch.Size([])
        )
        #
        # Standard variational inference.
        variational_strategy = make_orthogonal(self, input_dims, output_dims,num_inducing, batch_shape)
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        super().__init__(variational_strategy, input_dims, output_dims, Q)
        #
        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        #
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)
    #
    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def __init__(self, train_x_shape,layer):
        n = train_x_shape[0]
        self.n = n


class MultitaskDSPP(DSPP):
    def __init__(self, layer, train_x_shape, output_dims, inducing_points, num_inducing, hidden_dim, Q,mask):  
        n = train_x_shape[0]
        self.n = n
        hidden_layer = layer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            num_inducing=num_inducing,
            Q=Q,           
            inducing_points=inducing_points,
            mean_type='linear',
        )
        last_layer = layer(
            input_dims=hidden_layer.output_dims,
            output_dims=output_dims,
            num_inducing=num_inducing,
            Q=Q,            
            inducing_points=None,
            mean_type='constant',
        )   
        #
        super().__init__(Q)        
        if mask is not None:
            self.likelihood = MaskedMultitaskGaussianLikelihood(num_tasks=output_dims,rank=1,has_global_noise=False,has_task_noise=True)
        else:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=output_dims,rank=1,has_global_noise=False,has_task_noise=True)  
        #self.likelihood = MultitaskGaussianLikelihood(num_tasks=output_dims,rank=1,has_global_noise=False,has_task_noise=True)  
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer
    #
    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs, **kwargs)
        output = self.last_layer(hidden_rep1, **kwargs)
        return output
    #
    def predict(self, x_batch,y_batch):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls, Nls, Mss, Cos = [], [], [], [], [], []
            #for x_batch, y_batch in loader:
            preds = self.likelihood(self(x_batch, mean_input=x_batch))
            mus.append(preds.mean)
            variances.append(preds.variance)
            # Compute test log probability. The output of a DSPP is a weighted mixture of Q Gaussians,
            # with the Q weights specified by self.quad_weight_grid. The below code computes the log probability of each
            # test point under this mixture.
            # Step 1: Get log marginal for each Gaussian in the output mixture.
            base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))
            #Nlpd =  nlpd(self(x_batch),y_batch.to(torch.float))
            Msll =  msll(self(x_batch),y_batch)
            Cove =  cove(self(x_batch),y_batch)
            # Step 2: Weight each log marginal by its quadrature weight in log space.
            deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll
            #deep_batch_Nl = self.quad_weights.unsqueeze(-1) + Nlpd
            deep_batch_Ms = self.quad_weights.unsqueeze(-1) + Msll
            deep_batch_Co = self.quad_weights.unsqueeze(-1) + Cove
            # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
            batch_log_prob = deep_batch_ll.logsumexp(dim=0)
            #Nl = deep_batch_Nl.logsumexp(dim=0)
            Ms = deep_batch_Ms.logsumexp(dim=0)
            Co = deep_batch_Co.logsumexp(dim=0)
            lls.append(batch_log_prob.cpu())
            #Nls.append(Nl.cpu())
            Mss.append(Ms.cpu())
            Cos.append(Co.cpu())
    
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1),torch.cat(Mss, dim=-1), torch.cat(Cos, dim=-1)
