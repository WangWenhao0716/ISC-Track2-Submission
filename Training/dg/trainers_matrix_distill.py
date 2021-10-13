from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth, SoftTripletLoss
from .utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon=0.1).cuda()
        self.criterion_ce_1 = CrossEntropyLabelSmooth(num_classes, epsilon=0.1).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.w_ce = 10
        self.w_tri = 1
        self.w_ce_soft = 2
        self.w_ce_hard = 1
        self.temp = 2
        print("The weight for loss_ce is ", self.w_ce)
        print("The weight for loss_ce_soft is ", self.w_ce_soft)
        print("The weight for loss_tri is ", self.w_tri)
        print("The weight for temp is ", self.temp)

    def train(self, epoch, data_loader, optimizer, train_iters=200, print_freq=1):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_ce_1 = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()
        precisions_1 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out, s_cls_out_1= self.model(s_inputs)
            

            # backward main #
            loss_ce, loss_ce_1, loss_tr, prec1, prec1_1 = self._forward(s_features, s_cls_out, s_cls_out_1, targets)
            
            target_soft = self.softmax(s_cls_out/self.temp)
            log_probs = self.logsoftmax(s_cls_out_1/self.temp)
            loss_ce_soft = (- target_soft * log_probs).mean(0).sum()

            loss = self.w_ce * loss_ce + self.w_ce_hard * loss_ce_1 + self.w_ce_soft * loss_ce_soft + self.w_tri * loss_tr

            losses_ce.update(loss_ce.item())
            losses_ce_1.update(loss_ce_1.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)
            precisions_1.update(prec1_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'LR:{:.8f}\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_ce_1 {:.3f} ({:.3f})\t'
                      'Loss_ce_soft {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      'Prec_1 {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,optimizer.param_groups[0]["lr"],
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_ce_1.val, losses_ce_1.avg,
                              losses_ce_soft.val, losses_ce_soft.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg,
                              precisions_1.val, precisions_1.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, s_outputs_1, targets):
        s_features = s_features.cuda()
        s_outputs = s_outputs.cuda()
        s_outputs_1 = s_outputs_1.cuda()
        targets = targets.cuda()
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_ce_1 = self.criterion_ce_1(s_outputs_1, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]
        prec_1, = accuracy(s_outputs_1.data, targets.data)
        prec_1 = prec_1[0]

        return loss_ce, loss_ce_1, loss_tr, prec, prec_1


