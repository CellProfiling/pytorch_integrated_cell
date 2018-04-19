#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:04:13 2018

@author: trangle
"""
import torch
import importlib
import torch.optim as optim
import SimpleLogger
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.misc
import pickle
import importlib

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from imgToProjection import imgtoprojection

import pdb


def load_logger(parent_dir):
    
    struct_dir = parent_dir + os.sep + 'struct_model'
    #rename_opt_path(struct_dir)
        
    opt = pickle.load(open('{0}/opt.pkl'.format(struct_dir), "rb" ))
        
    columns = ('epoch', 'iter', 'reconLoss',)
    print_str = '[%d][%d] reconLoss: %.6f'
        
    if opt.nClasses > 0:
        columns += ('classLoss',)
        print_str += ' classLoss: %.6f'
            
    if opt.nRef > 0:
        columns += ('refLoss',)
        print_str += ' refLoss: %.6f'
        
    columns += ('minimaxEncDLoss', 'encDLoss', 'minimaxDecDLoss', 'decDLoss', 'time')
    print_str += ' mmEncD: %.6f encD: %.6f  mmDecD: %.6f decD: %.6f time: %.2f'
    logger = SimpleLogger.SimpleLogger(columns,  print_str)
    logger = pickle.load(open( '{0}/logger.pkl'.format(struct_dir), "rb" ))
    
    return logger



parent_dir = '/root/results/nucleoli_numt_latent16_1'
logger1=load_logger(parent_dir)
len(logger1.log['reconLoss'])

parent_dir = '/root/results/nucleoli_nu_latent16_1'
logger2=load_logger(parent_dir)

# Plotthing
def get_reconloss(logger, nepoch, head = True):
    
    if head == True:
        uepochs = range(0,nepoch)
        windows = int(len(logger.log['iter'])*nepoch/150)
        x = logger.log['iter'][0:windows]
        y = logger.log['reconLoss'][0:windows]
        epochs = np.floor(np.array(logger.log['epoch'][0:windows]))
        losses = np.array(logger.log['reconLoss'][0:windows])
        iters = np.array(logger.log['iter'][0:windows])
    else:        
        uepochs = range(nepoch,150) #consider to change it to opt.batch_size-1
        windows = int(len(logger.log['iter'])*nepoch/150)
        x = logger.log['iter'][-windows:]
        y = logger.log['reconLoss'][-windows:]
        epochs = np.floor(np.array(logger.log['epoch'][-windows:]))
        losses = np.array(logger.log['reconLoss'][-windows:])
        iters = np.array(logger.log['iter'][-windows:])
        
    epoch_losses = np.zeros(len(uepochs))
    epoch_iters = np.zeros(len(uepochs))
    i = 0
    for uepoch in uepochs:
        inds = np.equal(epochs, uepoch)
        loss = np.mean(losses[inds])
        epoch_losses[i] = loss
        epoch_iters[i] = uepoch#np.mean(iters[inds])
        i+=1
    mval = np.mean(losses)
    return x, epoch_iters, losses, epoch_losses, mval

iter1, epoch1,loss1,epochavgLoss1,mval1 = get_reconloss(logger1,150,head=True) #nu+mt
iter2, epoch2,loss2,epochavgLoss2, mval2 = get_reconloss(logger2,150,head=True) #nu

plt.figure()
plt.plot(epoch1, epochavgLoss1, color='r', label='reconLoss_nu+mt')
plt.plot(epoch2, epochavgLoss2, color='b', label='reconLoss_nu')
#plt.plot(epoch_iters, epoch_losses, color='darkorange', label='epoch avg')
plt.plot([np.min(epoch1), np.max(epoch1)], [mval1, mval1], color='orangered', linestyle=':', label='window avg_nu+mt')
plt.plot([np.min(epoch2), np.max(epoch2)], [mval2, mval2], color='cyan', linestyle=':', label='window avg_nu')
plt.legend()
plt.title('Short history')
plt.xlabel('epoch')
plt.ylabel('reconloss')
print('{0}/history_50lastepoch.png'.format(parent_dir))
plt.savefig('{0}/history_150epoch.png'.format(parent_dir), bbox_inches='tight')
plt.close()
