#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
#######

#######    
### Load the Model Parts
#######

import argparse

import SimpleLogger as SimpleLogger

import importlib
import numpy as np

import os
import pickle

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display
import time

import model_utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--batch_size', type=int, default=400, help='batch_size')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model' 

opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)

#######    
### Load REFERENCE MODEL
#######

opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

models, optimizers, _, _, opt = model_utils.load_model(opt.model_name, opt)

enc = models['enc']
dec = models['dec']

enc.train(False)
dec.train(False)

models = None
optimizers = None


print('Done loading model.')


#######    
### Main Loop
#######

import pdb
from aicsimage.io import omeTifWriter
from imgToProjection import imgtoprojection
from IPython.core.display import display
import PIL.Image
import matplotlib.pyplot as plt
import scipy.misc

import pandas as pd

gpu_id = 0

colormap = 'hsv'
colors = plt.get_cmap(colormap)(np.linspace(0, 1, 4))

px_size = [0.3873, 0.3873, 0.3873]

train_or_test_split = ['train', 'test']

img_paths_all = list()

save_parent = opt.save_dir + os.sep + 'images_out'
save_out_table = save_parent + os.sep + 'list_of_images.csv'

column_names = ['orig', 'recon'] + ['pred_' + name for name in dp.label_names] + ['train_or_test', 'orig_struct', 'img_index']

if not os.path.exists(save_parent):
    os.makedirs(save_parent)

def convert_image(img):
    img = img.data[0].cpu().numpy()
    img = np.transpose(img, (3, 0, 1, 2))
    
    return img

# For train or test
for train_or_test in train_or_test_split:
    ndat = dp.get_n_dat(train_or_test)
    # For each cell in the data split
    for i in range(0, ndat):
        print(str(i) + os.sep + str(ndat))
        
        
        img_index = dp.data[train_or_test]['inds'][i]
        img_class = dp.image_classes[img_index]
        img_name = os.path.basename(dp.get_image_paths([i], train_or_test)[0])[0:-3]
        
        save_dir = save_parent + os.sep + train_or_test + os.sep + img_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        #Load the image
        img_in = dp.get_images([i], train_or_test)
        img_in = Variable(img_in.cuda(gpu_id), volatile=True)
        
        #pass forward through the model
        z = enc(img_in)
        img_recon = dec(z)
        
        pred_imgs = list()
        img_paths = list()
        
        #print original images
        img_orig = convert_image(img_in)
        channel_names = ['memb', img_class, 'dna']
        img_name = save_dir + os.sep + 'img' + str(img_index) + '.ome.tif'
        with omeTifWriter.OmeTifWriter(img_name, overwrite_file=True) as w:
            w.save(img_orig, channel_names=channel_names, pixels_physical_size=px_size)
        
        pred_imgs.append(img_orig)
        img_paths.append(img_name)
        
        #print reconstructed images
        img_recon = convert_image(img_recon)
        channel_names_recon = ['memb_recon', img_class + '_recon', 'dna_recon']
        img_name = save_dir + os.sep + 'img' + str(img_index) + '_' + img_class + '-recon.ome.tif'
        with omeTifWriter.OmeTifWriter(img_name, overwrite_file=True) as w:
            w.save(img_recon, channel_names=channel_names_recon, pixels_physical_size=px_size)

        pred_imgs.append(img_recon)
        img_paths.append(img_name)
        channel_names += channel_names_recon
        
        #for each structure type
        for j in range(0, dp.get_n_classes()):
            pred_class_name = dp.label_names[j]
            
            img_name = save_dir + os.sep + 'img' + str(img_index) + '_' + img_class + '-pred_' + pred_class_name + '.ome.tif'
            
            #Set the class label in log(one-hot) form
            z[0].data[0] = torch.zeros(z[0].size()).cuda(gpu_id)
            z[0].data[0][j] = 1
            z[0].data[0] = (z[0].data[0] - 1) * 25
            
            #Reference variable is set as z[1]
            
            #Set the structure variation variable to most probable
            z[-1] = torch.zeros(z[-1].size()).cuda(gpu_id)
            
            #generate image with these settings
            img_recon = dec(z)
            
            #convert the image and get only the GFP channel
            img_recon = convert_image(img_recon)
            img_recon = np.expand_dims(img_recon[:,1,:,:],1)
            
            #save the gfp channel
            with omeTifWriter.OmeTifWriter(img_name, overwrite_file=True) as w:
                w.save(img_recon, channel_names=[pred_class_name + '_pred'], pixels_physical_size=px_size)
            
            channel_names.append(pred_class_name + ' pred')
            
            pred_imgs.append(img_recon)
            img_paths.append(img_name)
            
        img_paths += [train_or_test, img_class, img_index]
        img_paths_all.append(img_paths)
        
        pred_imgs_all = np.concatenate(pred_imgs,1)
        
        # save the all-channels image (orig, recon, and predicted structures)
        img_name = save_dir + os.sep + 'img' + str(img_index) + '_' + img_class + '-pred_all.ome.tif'
        with omeTifWriter.OmeTifWriter(img_name, overwrite_file=True) as w:
                w.save(pred_imgs_all, channel_names=channel_names, pixels_physical_size=px_size)
                
        images_proj = list()
        
        # save flat images
        img_in = convert_image(img_in)
        
        img = np.transpose(img_in, (1,0,2,3))
        img = imgtoprojection(img, proj_all=True, colors = colors, global_adjust=True)
        img = np.transpose(img, (1,2,0))
        
        images_proj.append(img)
        
        img = np.transpose(pred_imgs[1], (1,0,2,3))
        img = imgtoprojection(img, proj_all=True, colors = colors, global_adjust=True)
        img = np.transpose(img, (1,2,0))

        images_proj.append(img)
        
        for j in range(2, len(pred_imgs)):
            img = np.transpose(pred_imgs[j], (1,0,2,3))
            img = imgtoprojection(img, proj_all=True, global_adjust=True)
            img = np.transpose(img, (1,2,0))
            
            images_proj.append(img)
        
        images_proj = np.concatenate(images_proj,1)
        
        scipy.misc.imsave(save_dir + os.sep + 'img' + str(img_index) + '_' + img_class + '-pred_all.png', images_proj)

#save the list of all images
img_paths_all_df = pd.DataFrame(img_paths_all, columns=column_names);
img_paths_all_df.to_csv(save_out_table)

