# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
from vit_pytorch import ViT
# from timm.models import create_model
# from timm.models.vision_transformer import _cfg
from domainbed.lib.visiontransformer import *
from domainbed.lib.cross_visiontransformer import CrossVisionTransformer
from domainbed.lib.cvt import tiny_cvt,small_cvt
from domainbed.lib.CrossImageViT import CrossImageViT
import itertools
from prettytable import PrettyTable
import copy
import numpy as np
from torchvision.utils import save_image
from torchvision.utils import make_grid
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)

from domainbed import queue_var # for making queue: CorrespondenceSelfCross
queue_sz = queue_var.queue_sz

ALGORITHMS = [
    'ERM',
    'DeitSmall',
    'DeitTiny',
    'CVTTiny',
    'CrossImageVIT',
    'CrossImageVITSInf',
    'CrossImageVITSepCE',
    'CrossImageVITSepCE_SINF',
    'CrossImageVIT_self_SepCE_SINF',
    'CrossImageVITDeit',
    'ERMBrainstorm',
    'JustTransformer',
    'CorrespondenceSelfCross',
    'Correspondence',
    'CorrespondenceSelf',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'Testing'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        print("num_domains:",num_domains)
        
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class DeitSmall(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_small_patch16_224', pretrained=True, source='local')    
        self.network=deit_small_patch16_224(pretrained=False) 
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            eps=self.hparams['eps']
        )
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)
   
class DeitTiny(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitTiny, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_Tiny_patch16_224', pretrained=True, source='local')    
        self.network=deit_tiny_patch16_224(pretrained=False) 
        self.network.head = nn.Linear(192, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
        )

class CVTSmall(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CVTSmall, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_Tiny_patch16_224', pretrained=True, source='local')    
        self.network=small_cvt(pretrained=False) 
        # print(self.network)
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)[-1]

class CVTTiny(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CVTTiny, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_Tiny_patch16_224', pretrained=True, source='local')    
        self.network=tiny_cvt(pretrained=False) 
        # print(self.network)
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)[-1]



class CrossImageVIT(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVIT, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=4,
                im_enc_depth=2,cross_attn_depth=2,num_heads=8, representation_size=None, distilled=False,
                 drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 8,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred=self.predictTrain(cross_learning_data)
        loss = F.cross_entropy(pred, cross_learning_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x]*self.num_domains)
    def predictTrain(self, x):
        return self.network(x)


class CrossImageVITSInf(ERM):
    """
    cross image vit with single image inference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVITSInf, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=1,
                im_enc_depth=12,cross_attn_depth=3,num_heads=6, representation_size=None, distilled=False,
                 drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 6,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred=self.predictTrain(cross_learning_data)
        loss = F.cross_entropy(pred, cross_learning_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x])
    def predictTrain(self, x):
        return self.network(x)

class CrossImageVITSepCE(ERM):
    """
    cross image vit with single image inference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVITSepCE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=1,
                im_enc_depth=12,cross_attn_depth=3,num_heads=6, representation_size=None, distilled=False,
                 drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 6,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred=self.predictTrain(cross_learning_data)
        loss = 0
        for dom_n in range (self.num_domains):
            loss+= F.cross_entropy(pred[dom_n], cross_learning_labels)
        loss=loss*1.0/self.num_domains
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x]*self.num_domains)
    def predictTrain(self, x):
        return self.network(x,return_list=True)


class CrossImageVITSepCE_SINF(ERM):

    """
    cross image vit with single image inference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVITSepCE_SINF, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=4,
            im_enc_depth=2,cross_attn_depth=2,num_heads=8, representation_size=None, distilled=False,
            drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 8,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64)
        # printNetworkParams(self.network)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred=self.predictTrain(cross_learning_data)
        loss = 0

        for dom_n in range (self.num_domains):
            loss+= F.cross_entropy(pred[dom_n], cross_learning_labels)
        loss=loss*1.0/self.num_domains


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x])
    def predictTrain(self, x):
        return self.network(x,return_list=True)

class CrossImageVIT_self_SepCE_SINF(ERM):

    """
    cross image vit with single image inference and seperated CE for both final output & self attn out 
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVIT_self_SepCE_SINF, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=1,
            im_enc_depth=8,cross_attn_depth=4,num_heads=6, representation_size=None, distilled=False,
            drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 6,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64,return_self=True)
        printNetworkParams(self.network)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred,pred_only_self=self.predictTrain(cross_learning_data)
        loss = 0

        for dom_n in range (self.num_domains):
            loss+= F.cross_entropy(pred[dom_n], cross_learning_labels)+F.cross_entropy(pred_only_self[dom_n], cross_learning_labels)
            
        loss=loss*1.0/self.num_domains


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x])
    def predictTrain(self, x):
        return self.network(x,return_list=True)

class CrossImageVITDeit(ERM):
    """
    cross image vit with single image inference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVITDeit, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=4,
                im_enc_depth=3,cross_attn_depth=2,num_heads=6, representation_size=None, distilled=False,
                 drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 8,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64,nocross=True)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred=self.predictTrain(cross_learning_data)
        loss = 0
        for dom_n in range (len(pred)):
            loss+= F.cross_entropy(pred[dom_n], cross_learning_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x]*self.num_domains)
    def predictTrain(self, x):
        return self.network(x,return_list=True)
class DeitSmallDtest(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmallDtest, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_small_patch16_224', pretrained=True, source='local')    
        self.network=deit_small_patch16_224(pretrained=True) 
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            eps=self.hparams['eps']
        )
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # loss = F.cross_entropy(self.predict(all_x), all_y)
        
        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        all_labels=[]
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
                # all_labels+=label_tensor
        cross_learning_data1=[]
        # # cross_learning_data2=[]
        cross_learning_labels=[]
        # domain_nums=list(range(ndomains))
        # combinations=itertools.combinations(domain_nums, 2)
        
        for i in range(queue_sz):
            for cls in range(nclass):
                for j in range(3):
                    cross_learning_data1.append(train_queues[cls][j][i])
                    cross_learning_labels.append(cls)
                # cross_learning_data2.append(train_queues[cls][subset[1]][i])
                
        
        
        cross_learning_data1=torch.stack(cross_learning_data1)
        # # cross_learning_data2=torch.stack(cross_learning_data2)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        # # crossLoss=F.cross_entropy(self.crossnet(cross_learning_data1,cross_learning_data2), cross_learning_labels)
        # # totloss=loss+crossLoss
        # print(cross_learning_data1.shape)
        # print(cross_learning_labels)
        pred=self.predict(cross_learning_data1)
        loss = F.cross_entropy(pred, cross_learning_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)
   

class CorrespondenceSelfCross(Algorithm):
    """
    Self and cross correspondence 
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CorrespondenceSelfCross, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        checkpoint=torch.load("/home/computervision1/Sanoojan/DomainBedS/domainbed/pretrained/best_checkpoint.pth")
        
        self.CrossAttention=CrossVisionTransformer(img_size=7, in_chans=512, patch_size=1, num_classes=num_classes, embed_dim=512, depth=2,
                 num_heads=2, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.)

        self.SelfAttention = ViT(
            image_size = 7,
            patch_size = 1,
            num_classes = 1000,
            dim = 512,
            depth = 2,
            channels=512,
            heads = 2,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        
        # self.featurizer = torchvision.models.resnet50(pretrained=True)

        self.featurizer =nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
    
        # nn.Sequential(*list(networks.Featurizer(input_shape, self.hparams).ResNet.children())[:-1])
        # print(self.featurizer)
        # print(*list(networks.Featurizer(input_shape, self.hparams).children())[:-2])
        self.classifier = networks.Classifier(
            1000,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.SelfAttention)
        self.network.load_state_dict(checkpoint['model'])
        self.network.eval()
        self.network=nn.Sequential(self.network,self.classifier)
        self.crossnet=nn.Sequential(self.featurizer,self.CrossAttention)
        # print(self.network)
        # print("params",self.network.parameters())
        self.optimizer = torch.optim.AdamW(list(self.network.parameters()) + list(self.crossnet.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams['weight_decay']
        # )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        
        # print(self.featurizer(all_x).shape)
        
        
        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
        cross_learning_data1=[]
        cross_learning_data2=[]
        cross_learning_labels=[]
        domain_nums=list(range(ndomains))
        combinations=itertools.combinations(domain_nums, 2)
        for subset in combinations:
            for i in range(queue_sz):
                for cls in range(nclass):
                    cross_learning_data1.append(train_queues[cls][subset[0]][i])
                    cross_learning_data2.append(train_queues[cls][subset[1]][i])
                    cross_learning_labels.append(cls)
        
        # print(train_queues)
        cross_learning_data1=torch.stack(cross_learning_data1)
        cross_learning_data2=torch.stack(cross_learning_data2)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        crossLoss=F.cross_entropy(self.crossnet(cross_learning_data1,cross_learning_data2), cross_learning_labels)
        totloss=loss+crossLoss
        
        self.optimizer.zero_grad()
        totloss.backward()
        self.optimizer.step()
        return {'loss': totloss.item()}
        
        # all_x=None
        # all_y=None

        
        # for id_d in range(num_domains): # loop over domains
        #     mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
        #     # indices of those egs from domain id_d, whose class label is id_c
        #     label_tensor=minibatches[id_d][1][mb_ids] # labels
        #     if mb_ids.size(0)==0:
        #         #print('class has no element')
        #         continue
        #     data_tensor=minibatches[id_d][0][mb_ids] # data

        # print(len(minibatches))
        totalLoss=0
        domainlabels=[]
        for domain_num in range(len(minibatches)):

            all_x = torch.cat([x for x,y in [minibatches[domain_num]]])
            all_y = torch.cat([y for x,y in [minibatches[domain_num]]])
            # print(type(all_y))
            loss = F.cross_entropy(self.predictTrain(all_x,domain_num), all_y)
            # print("domain:",domain_num," ,all_x_size:",all_x.size())
            # print("domain:",domain_num," ,all_y_size:",all_y.size())
            domainlabels+=[domain_num]*len(all_y)
            self.optimizer[domain_num].zero_grad()
            loss.backward()
            self.optimizer[domain_num].step()
            totalLoss+=loss.item()

        domlabels=torch.LongTensor(domainlabels)
        domlabels=domlabels.cuda()
        all_x = torch.cat([x for x,y in minibatches])
        loss = F.cross_entropy(self.predictDomain(all_x), domlabels)
        self.DDNoptimizer.zero_grad()
        loss.backward()
        self.DDNoptimizer.step()
        return {'loss': totalLoss}

    def predict(self, x):
        return self.network(x)
        # print("predict has been called terminate.......")
        domainprediction=self.DDN(x)
        dompredict=torch.argmax(domainprediction, dim=1)
        # print("domain predictions",dompredict)
        currdomain=dompredict[0].item()
        # print(dompredict[0].item(),"first one")
        # print(x[0].shape, "first shape")
        # print(self.network[0](x[0].unsqueeze(0)))

        return torch.cat([self.network[dompredict[i].item()](x[i].unsqueeze(0)) for i in dompredict])
        # return np.array([self.network[dompredict[i].item()](x[i].unsqueeze(0))[0].cpu() for i in dompredict])

        # for net in self.network:
        #     predictions.append(net(x))
            
        # return self.network[currdomain](x)
    
    def predictDomain(self,x):
        return self.DDN(x)
        # return self.v(x)

    def predictTrain(self,x,domain):
        if torch.cuda.is_available():
            self.network[domain].cuda()
        # print(self.network[domain](x))
        return self.network[domain](x)


class CorrespondenceSelf(Algorithm):
    """
    Creating a weighted model from seperately trained models from ERM on seperate doains (ERMWeightedModel)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CorrespondenceSelf, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        checkpoint=torch.load("/home/computervision1/Sanoojan/DomainBedS/domainbed/pretrained/best_checkpoint.pth")
        
        self.SelfAttention = ViT(
            image_size = 7,
            patch_size = 1,
            num_classes = 1000,
            dim = 512,
            depth = 2,
            channels=512,
            heads = 2,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.CrossAttention=ViT(
            image_size = 224,
            patch_size = 14,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        # self.featurizer = torchvision.models.resnet50(pretrained=True)

        self.featurizer =nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
    
        # nn.Sequential(*list(networks.Featurizer(input_shape, self.hparams).ResNet.children())[:-1])
        # print(self.featurizer)
        # print(*list(networks.Featurizer(input_shape, self.hparams).children())[:-2])
        self.classifier = networks.Classifier(
            1000,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.SelfAttention)
        self.network.load_state_dict(checkpoint['model'])
        self.network.eval()
        self.network=nn.Sequential(self.network,self.classifier)
        # print(self.network)
        # print("params",self.network.parameters())
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        # print(self.featurizer(all_x).shape)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

        # train_queues = queue_var.train_queues
        # nclass=len(train_queues)
        # ndomains=len(train_queues[0])
        
        # all_x=None
        # all_y=None

        
        # for id_d in range(num_domains): # loop over domains
        #     mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
        #     # indices of those egs from domain id_d, whose class label is id_c
        #     label_tensor=minibatches[id_d][1][mb_ids] # labels
        #     if mb_ids.size(0)==0:
        #         #print('class has no element')
        #         continue
        #     data_tensor=minibatches[id_d][0][mb_ids] # data

        # print(len(minibatches))
        totalLoss=0
        domainlabels=[]
        for domain_num in range(len(minibatches)):

            all_x = torch.cat([x for x,y in [minibatches[domain_num]]])
            all_y = torch.cat([y for x,y in [minibatches[domain_num]]])
            # print(type(all_y))
            loss = F.cross_entropy(self.predictTrain(all_x,domain_num), all_y)
            # print("domain:",domain_num," ,all_x_size:",all_x.size())
            # print("domain:",domain_num," ,all_y_size:",all_y.size())
            domainlabels+=[domain_num]*len(all_y)
            self.optimizer[domain_num].zero_grad()
            loss.backward()
            self.optimizer[domain_num].step()
            totalLoss+=loss.item()

        domlabels=torch.LongTensor(domainlabels)
        domlabels=domlabels.cuda()
        all_x = torch.cat([x for x,y in minibatches])
        loss = F.cross_entropy(self.predictDomain(all_x), domlabels)
        self.DDNoptimizer.zero_grad()
        loss.backward()
        self.DDNoptimizer.step()
        return {'loss': totalLoss}

    def predict(self, x):
        return self.network(x)
        # print("predict has been called terminate.......")
        domainprediction=self.DDN(x)
        dompredict=torch.argmax(domainprediction, dim=1)
        # print("domain predictions",dompredict)
        currdomain=dompredict[0].item()
        # print(dompredict[0].item(),"first one")
        # print(x[0].shape, "first shape")
        # print(self.network[0](x[0].unsqueeze(0)))

        return torch.cat([self.network[dompredict[i].item()](x[i].unsqueeze(0)) for i in dompredict])
        # return np.array([self.network[dompredict[i].item()](x[i].unsqueeze(0))[0].cpu() for i in dompredict])

        # for net in self.network:
        #     predictions.append(net(x))
            
        # return self.network[currdomain](x)
    
    def predictDomain(self,x):
        return self.DDN(x)
        # return self.v(x)

    def predictTrain(self,x,domain):
        if torch.cuda.is_available():
            self.network[domain].cuda()
        # print(self.network[domain](x))
        return self.network[domain](x)

class Correspondence(Algorithm):
    """
    correspondence
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Correspondence, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.SelfAttention = ViT(
            image_size = 7,
            patch_size = 1,
            num_classes = num_classes,
            dim = 512,
            depth = 2,
            channels=512,
            heads = 2,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.CrossAttention=ViT(
            image_size = 224,
            patch_size = 14,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        # self.featurizer = torchvision.models.resnet50(pretrained=True)

        self.featurizer =nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
        # nn.Sequential(*list(networks.Featurizer(input_shape, self.hparams).ResNet.children())[:-1])
        # print(self.featurizer)
        # print(*list(networks.Featurizer(input_shape, self.hparams).children())[:-2])
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     num_classes,
        #     self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.SelfAttention)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        # print(self.featurizer(all_x).shape)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

        # train_queues = queue_var.train_queues
        # nclass=len(train_queues)
        # ndomains=len(train_queues[0])
        
        # all_x=None
        # all_y=None

        
        # for id_d in range(num_domains): # loop over domains
        #     mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
        #     # indices of those egs from domain id_d, whose class label is id_c
        #     label_tensor=minibatches[id_d][1][mb_ids] # labels
        #     if mb_ids.size(0)==0:
        #         #print('class has no element')
        #         continue
        #     data_tensor=minibatches[id_d][0][mb_ids] # data

        # print(len(minibatches))
        totalLoss=0
        domainlabels=[]
        for domain_num in range(len(minibatches)):

            all_x = torch.cat([x for x,y in [minibatches[domain_num]]])
            all_y = torch.cat([y for x,y in [minibatches[domain_num]]])
            # print(type(all_y))
            loss = F.cross_entropy(self.predictTrain(all_x,domain_num), all_y)
            # print("domain:",domain_num," ,all_x_size:",all_x.size())
            # print("domain:",domain_num," ,all_y_size:",all_y.size())
            domainlabels+=[domain_num]*len(all_y)
            self.optimizer[domain_num].zero_grad()
            loss.backward()
            self.optimizer[domain_num].step()
            totalLoss+=loss.item()

        domlabels=torch.LongTensor(domainlabels)
        domlabels=domlabels.cuda()
        all_x = torch.cat([x for x,y in minibatches])
        loss = F.cross_entropy(self.predictDomain(all_x), domlabels)
        self.DDNoptimizer.zero_grad()
        loss.backward()
        self.DDNoptimizer.step()
        return {'loss': totalLoss}

    def predict(self, x):
        return self.network(x)
        # print("predict has been called terminate.......")
        domainprediction=self.DDN(x)
        dompredict=torch.argmax(domainprediction, dim=1)
        # print("domain predictions",dompredict)
        currdomain=dompredict[0].item()
        # print(dompredict[0].item(),"first one")
        # print(x[0].shape, "first shape")
        # print(self.network[0](x[0].unsqueeze(0)))

        return torch.cat([self.network[dompredict[i].item()](x[i].unsqueeze(0)) for i in dompredict])
        # return np.array([self.network[dompredict[i].item()](x[i].unsqueeze(0))[0].cpu() for i in dompredict])

        # for net in self.network:
        #     predictions.append(net(x))
            
        # return self.network[currdomain](x)
    
    def predictDomain(self,x):
        return self.DDN(x)
        # return self.v(x)

    def predictTrain(self,x,domain):
        if torch.cuda.is_available():
            self.network[domain].cuda()
        # print(self.network[domain](x))
        return self.network[domain](x)

class JustTransformer(Algorithm):
    """
    Creating a weighted model from seperately trained models from ERM on seperate doains (ERMWeightedModel)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(JustTransformer, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.SelfAttention = ViT(
            image_size = 224,
            patch_size = 14,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.CrossAttention=ViT(
            image_size = 224,
            patch_size = 14,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.SelfAttention.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

        # train_queues = queue_var.train_queues
        # nclass=len(train_queues)
        # ndomains=len(train_queues[0])
        
        # all_x=None
        # all_y=None

        
        # for id_d in range(num_domains): # loop over domains
        #     mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
        #     # indices of those egs from domain id_d, whose class label is id_c
        #     label_tensor=minibatches[id_d][1][mb_ids] # labels
        #     if mb_ids.size(0)==0:
        #         #print('class has no element')
        #         continue
        #     data_tensor=minibatches[id_d][0][mb_ids] # data

        # print(len(minibatches))
        totalLoss=0
        domainlabels=[]
        for domain_num in range(len(minibatches)):

            all_x = torch.cat([x for x,y in [minibatches[domain_num]]])
            all_y = torch.cat([y for x,y in [minibatches[domain_num]]])
            # print(type(all_y))
            loss = F.cross_entropy(self.predictTrain(all_x,domain_num), all_y)
            # print("domain:",domain_num," ,all_x_size:",all_x.size())
            # print("domain:",domain_num," ,all_y_size:",all_y.size())
            domainlabels+=[domain_num]*len(all_y)
            self.optimizer[domain_num].zero_grad()
            loss.backward()
            self.optimizer[domain_num].step()
            totalLoss+=loss.item()

        domlabels=torch.LongTensor(domainlabels)
        domlabels=domlabels.cuda()
        all_x = torch.cat([x for x,y in minibatches])
        loss = F.cross_entropy(self.predictDomain(all_x), domlabels)
        self.DDNoptimizer.zero_grad()
        loss.backward()
        self.DDNoptimizer.step()
        return {'loss': totalLoss}

    def predict(self, x):
        return self.SelfAttention(x)
        # print("predict has been called terminate.......")
        domainprediction=self.DDN(x)
        dompredict=torch.argmax(domainprediction, dim=1)
        # print("domain predictions",dompredict)
        currdomain=dompredict[0].item()
        # print(dompredict[0].item(),"first one")
        # print(x[0].shape, "first shape")
        # print(self.network[0](x[0].unsqueeze(0)))

        return torch.cat([self.network[dompredict[i].item()](x[i].unsqueeze(0)) for i in dompredict])
        # return np.array([self.network[dompredict[i].item()](x[i].unsqueeze(0))[0].cpu() for i in dompredict])

        # for net in self.network:
        #     predictions.append(net(x))
            
        # return self.network[currdomain](x)
    
    def predictDomain(self,x):
        return self.DDN(x)
        # return self.v(x)

    def predictTrain(self,x,domain):
        if torch.cuda.is_available():
            self.network[domain].cuda()
        # print(self.network[domain](x))
        return self.network[domain](x)

class ERMBrainstorm(Algorithm):
    """
    Brainstorming model related to test domain Empirical (Risk Minimization with Brainstorm)  (ERMBrainstorm)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERMBrainstorm, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = [networks.Featurizer(input_shape, self.hparams) for i in range(num_domains)]

        self.DDNfeaturizer=networks.Featurizer(input_shape, self.hparams)
        self.DDNClassifier=networks.Classifier(
            self.DDNfeaturizer.n_outputs,
            num_domains,
            self.hparams['nonlinear_classifier'])
        self.DDN=nn.Sequential(self.DDNfeaturizer, self.DDNClassifier)
        self.DDNoptimizer=torch.optim.Adam(
            self.DDN.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.classifier = [networks.Classifier(
            self.featurizer[i].n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])  for i in range(num_domains)]

        self.network = [nn.Sequential(self.featurizer[i], self.classifier[i]) for i in range(num_domains) ]
        self.optimizer = [torch.optim.Adam(
            self.network[i].parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        ) for i in range(num_domains)]
    
    def update(self, minibatches, unlabeled=None):
        # train_queues = queue_var.train_queues
        # nclass=len(train_queues)
        # ndomains=len(train_queues[0])
        
        # all_x=None
        # all_y=None

        
        # for id_d in range(num_domains): # loop over domains
        #     mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
        #     # indices of those egs from domain id_d, whose class label is id_c
        #     label_tensor=minibatches[id_d][1][mb_ids] # labels
        #     if mb_ids.size(0)==0:
        #         #print('class has no element')
        #         continue
        #     data_tensor=minibatches[id_d][0][mb_ids] # data

        # print(len(minibatches))
        totalLoss=0
        domainlabels=[]
        for domain_num in range(len(minibatches)):
            
            all_x = torch.cat([x for x,y in [minibatches[domain_num]]])
            all_y = torch.cat([y for x,y in [minibatches[domain_num]]])
            # for i in range(5):
            #     torchvision.utils.save_image(all_x[i],'/home/computervision1/Sanoojan/DomainBed/domainbed/outputs/imageOuts/'+str(domain_num)+'/'+str(i)+'.png')
            # print(type(all_y))
            loss = F.cross_entropy(self.predictTrain(all_x,domain_num), all_y)
            # print("domain:",domain_num," ,all_x_size:",all_x.size())
            # print("domain:",domain_num," ,all_y_size:",all_y.size())
            domainlabels+=[domain_num]*len(all_y)
            self.optimizer[domain_num].zero_grad()
            loss.backward()
            self.optimizer[domain_num].step()
            totalLoss+=loss.item()

        domlabels=torch.LongTensor(domainlabels)
        domlabels=domlabels.cuda()
        all_x = torch.cat([x for x,y in minibatches])
        loss = F.cross_entropy(self.predictDomain(all_x), domlabels)
        self.DDNoptimizer.zero_grad()
        loss.backward()
        self.DDNoptimizer.step()
        return {'loss': totalLoss}

    def predict(self, x):
        # print("predict has been called terminate.......")
        domainprediction=self.DDN(x)
        dompredict=torch.argmax(domainprediction, dim=1)
        # print("Domain prediction:",dompredict)
        currdomain=dompredict[0].item()
        # print(dompredict[0].item(),"first one")
        # print(x[0].shape, "first shape")
        # print(self.network[0](x[0].unsqueeze(0)))

        # try to group images according to domains
        return torch.cat([self.network[dompredict[i].item()](x[i].unsqueeze(0)) for i in range(len(dompredict))])
        # return np.array([self.network[dompredict[i].item()](x[i].unsqueeze(0))[0].cpu() for i in dompredict])

        # for net in self.network:
        #     predictions.append(net(x))
            
        # return self.network[currdomain](x)
    
    def predictDomain(self,x):
        return self.DDN(x)

    def predictTrain(self,x,domain):
        if torch.cuda.is_available():
            self.network[domain].cuda()
        # print(self.network[domain](x))
        return self.network[domain](x)



class Testing(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Testing, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # print(self.featurizer)
        fname="/home/computervision1/Sanoojan/DomainBedS/domainbed/outputs/save_mod_test_deit/model.pkl"
        try:
            self.network=load_model(fname).network
        except:
            self.network=load_model(fname).network_original
        
        self.network.eval()
        # print(len(self.network.blocks))
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     num_classes,
        #     self.hparams['nonlinear_classifier'])

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        
        return self.network(x)

def load_model(fname):
    dump = torch.load(fname)
    algorithm_class = get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        dump["model_num_domains"],
        dump["model_hparams"])
    algorithm.load_state_dict(dump["model_dict"])
    return algorithm

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def printNetworkParams(net):
    # print("network1====",net)
    # count_parameters(net)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("pytorch_total_params:",pytorch_total_params)
    print("pytorch_total_trainable_params:",pytorch_total_trainable_params)