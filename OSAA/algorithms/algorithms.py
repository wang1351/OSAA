import torch
import torch.nn as nn
import numpy as np
import pdb
import random

from models.models import *
from models.loss import *


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]



class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

class DISTANT(Algorithm):

    def __init__(self, backbone_fe, backbone_decoder,configs, hparams, device):
        super(DISTANT, self).__init__(configs)        
        self.clasweight = hparams["clasweight"]
        self.domainweight = hparams["domainweight"]

        self.threshold_s = hparams["threshold_s"]*2 #need *2, after normalization of two [0,1], overall is [0, 2]
        self.threshold_i = hparams["threshold_i"]*2
        
        self.domain_classifier = Discriminator_(configs)
        
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.feature_extractor  = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.decoder = backbone_decoder(configs)
        
        self.network = nn.Sequential(self.feature_extractor,self.decoder, self.classifier)
        #pdb.set_trace()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.criterion_cond = ConditionalEntropyLoss().cuda()
        
    def update(self, src_x, src_y, inter_x, trg_x, ep, len_dataloader):
        src_h, src_indlist = self.feature_extractor(src_x)
        inter_h, inter_indlist = self.feature_extractor(inter_x)
        trg_h, trg_indlist = self.feature_extractor(trg_x)  
       
        
        src_hat = self.decoder(src_h.unsqueeze(2), src_indlist)
        inter_hat = self.decoder(inter_h.unsqueeze(2), inter_indlist)
        trg_hat = self.decoder(trg_h.unsqueeze(2), trg_indlist)
        
        src_recloss = (src_hat - src_x).pow(2).sum(-1).sqrt().squeeze(1)
        inter_recloss = (inter_hat - inter_x).pow(2).sum(-1).sqrt().squeeze(1)
        trg_recloss = (trg_hat - trg_x).pow(2).sum(-1).sqrt().squeeze(1)
        
        src_pred = self.classifier(src_h)
        inter_pred = self.classifier(inter_h)
        trg_pred = self.classifier(trg_h)
        
        feat_concat = torch.cat((src_h, trg_h), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        domain_label_src = torch.zeros(len(src_x)).long().cuda()
        domain_label_trg = torch.ones(len(trg_x)).long().cuda()
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
        
                # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        
                # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()
        
  # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().cuda()
        domain_label_trg = torch.ones(len(trg_x)).long().cuda()
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        # loss of domain discriminator according to fake labels

        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        src_clasloss = self.cross_entropy(src_pred, src_y)
        src_clasloss_threshold = - torch.mul(nn.functional.softmax(src_pred),nn.functional.one_hot(src_y,3)).sum(-1)

        inter_prob = nn.functional.softmax(inter_pred, dim=1)
        inter_entropy = - torch.mul(inter_prob,torch.log(inter_prob)).sum(-1)
        
        if ep==1:
            vS = torch.ones(len(src_y),).cuda()
            vI = torch.zeros(len(src_y),).cuda()
           
        else: #dominated by recloss, need normalization!
            svalue = (src_recloss - src_recloss.min()) /(src_recloss.max() - src_recloss.min()) +  (src_clasloss_threshold - src_clasloss_threshold.min()) /(src_clasloss_threshold.max() - src_clasloss_threshold.min())
            ivalue = (inter_recloss - inter_recloss.min()) /(inter_recloss.max() - inter_recloss.min()) +  (inter_entropy - inter_entropy.min()) /(inter_entropy.max() - inter_entropy.min())
            vS = torch.where(svalue < self.threshold_s, 1, 0)
            vI = torch.where(ivalue < self.threshold_i, 1, 0) 

        R_ = - self.threshold_s * vS.sum() - self.threshold_i * vI.sum()

        J1 = R_  + torch.mul(vS, src_recloss).sum() + torch.mul(vI, inter_recloss).sum() + trg_recloss.sum()
        J2 = torch.mul(vS, src_clasloss).sum() + torch.mul(vI, inter_entropy).sum()
        
        if not J2 > 0:
            #pdb.set_trace()
            loss = J1
        else:
            loss = J1  + self.clasweight * J2# + self.mmd_w * mmd_loss
        loss = loss + self.clasweight*self.criterion_cond(trg_pred)  + self.domainweight*domain_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'J1_loss': J1.item(), 'J2_loss': J2.item(),'domain_loss': domain_loss.item()}


    
    
