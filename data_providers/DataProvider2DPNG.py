import glob
import os
import numpy as np
from scipy import misc
from natsort import natsorted
from PIL import Image
import torch

import pdb

from data_providers.DataProviderABC import DataProviderABC


class DataProvider(DataProviderABC):
    
    def __init__(self, image_parent, opts={}):
        self.data = {}
        
        opts_default = {'rotate': False, 'hold_out': 1/10, 'verbose': True, 'channelInds': [0,1,2], 'h5_file': True}
        
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts

        # assume file structure is <image_parent>/<class directory>/*.png
        image_dirs = natsorted(glob.glob(image_parent + os.sep + '*'))

        image_paths = list()
        for image_dir in image_dirs:
            image_paths += natsorted(glob.glob(image_dir + os.sep + '*.png'))

        image_classes = [os.path.basename(os.path.dirname(image_path)) for image_path in image_paths]

        self.image_paths = image_paths
        self.image_classes = image_classes
        
        nimgs = len(image_paths)

        [label_names, labels] = np.unique(image_classes, return_inverse=True)
        self.label_names = label_names
        
        onehot = np.zeros((nimgs, np.max(labels)+1))
        onehot[np.arange(nimgs), labels] = 1
        
        self.labels = labels
        self.labels_onehot = onehot
    
        rand_inds = np.random.permutation(nimgs)
        
        ntest = int(np.round(nimgs*opts['hold_out']))
        
        self.data['test'] = {}
        self.data['test']['inds'] = rand_inds[0:ntest+1]
        
        self.data['train'] = {}
        self.data['train']['inds'] = rand_inds[ntest+2:-1]
        
        images = list()
        c = 0
        for image_path in image_paths:
            img = self.open_image(image_path)
            images.append(img)
            c += 1
            
        print("Image load " + str(c) + '/' + str(len(image_paths)))
        if self.opts['verbose']:
            print('{0} images are present in shape {1}'.format(len(image_paths),img.shape))
        print(images.dtype)
        images = np.concatenate(images, 0)
        breakme
        images = torch.FloatTensor(images)
        self.all_images = images
        self.imsize = self.all_images[0].size()
        print(self.imsize)
        

    def open_image(self, image_path):
        image = Image.open(image_path)
        img = np.asarray(image)
        img = np.float32(img)/255
        img = np.transpose(img, [2,0,1])
        img = np.expand_dims(img, 0)

        return img

    def get_n_dat(self, train_or_test = 'train'):
        return len(self.data[train_or_test]['inds'])

#     def get_n_train(self):
#         return len(self.data['train']['inds'])
    
#     def get_n_test(self):
#         return len(self.data['test']['inds'])

    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
    def get_images(self, inds, train_or_test):
        dims = list(self.all_images[0].size())
        dims[0] = len(self.opts['channelInds'])
        
        dims.insert(0, len(inds))
        print(dims)
        images = torch.zeros(tuple(dims))

        c = 0
        for i in inds:
            image = self.all_images[self.data[train_or_test]['inds'][i]]
            images[c] = image.index_select(0, torch.LongTensor(self.opts['channelInds'])).clone()
            c += 1

        if self.opts['verbose']:
            print('{0}/{1} files are loaded'.format(c,len(self.image_paths)))
        
        return images
    
    def get_classes(self, inds, train_or_test, index_or_onehot = 'index'):
        
        if index_or_onehot == 'index':
            labels = self.labels[self.data[train_or_test]['inds'][inds]]
            labels = torch.LongTensor(labels)
        else:
            labels = np.zeros([len(inds), self.get_n_classes()])
            c = 0
            for i in inds:
                labels[c] = self.labels_onehot[self.data[train_or_test]['inds'][i]]
                
                c += 1
            
            labels = torch.from_numpy(labels).long()
            
        return labels
    
    def get_rand_images(self, batsize, train_or_test):
        ndat = self.data[train_or_test]['inds']
        
        inds = np.random.choice(ndat, batsize)
        
        return self.get_images(inds, train_or_test)

    def get_image_paths(self,inds_tt,train_or_test):
        image_paths = list()
        for image_dir in image_dirs:
            image_paths += natsorted(glob.glob(image_dir + os.sep + '*.h5'))
        return image_paths    
