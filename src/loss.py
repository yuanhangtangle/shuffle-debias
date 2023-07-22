import torch
import torch.nn as nn
import torch.nn.functional as F
'''
The implementation of the 2 loss are adapted from:
https://github.com/rabeehk/robust-nli/blob/master/src/losses.py
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, size_average=True, ensemble_training=False, aggregate_ensemble="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ensemble_training = ensemble_training
        self.aggregate_ensemble=aggregate_ensemble

    def compute_probs(self, inputs, targets):
        prob_dist = F.softmax(inputs, dim=1)
        pt = prob_dist.gather(1, targets)
        return pt 

    def aggregate(self, p1, p2, operation):
        if self.aggregate_ensemble == "mean":
            result = (p1+p2)/2
            return result
        elif self.aggregate_ensemble == "multiply":
            result = p1*p2
            return result
        else:
            assert NotImplementedError("Operation ", operation, "is not implemented.")

    def forward(self, inputs, targets, inputs_adv=None, second_inputs_adv=None, inputs_adv_is_prob = False):
        targets = targets.view(-1, 1)
        norm = 0.0
        if inputs_adv is None:
            inputs_adv = inputs
        pt = self.compute_probs(inputs, targets)
        if inputs_adv_is_prob:
            pt_scale = inputs_adv.gather(1, targets)
        else:
            pt_scale = self.compute_probs(inputs_adv, targets)
        if self.ensemble_training:
            pt_scale_second = self.compute_probs(second_inputs_adv, targets)
            if self.aggregate_ensemble in ["mean", "multiply"]:
                pt_scale_total = self.aggregate(pt_scale, pt_scale_second, "mean")
                batch_loss = -self.alpha * (torch.pow((1 - pt_scale_total), self.gamma)) * torch.log(pt)
        else:
            batch_loss = -self.alpha * (torch.pow((1 - pt_scale), self.gamma)) * torch.log(pt)
        norm += self.alpha * (torch.pow((1 - pt_scale), self.gamma))

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class POELoss(nn.Module):
    """Implements the product of expert loss."""
    def __init__(self, size_average=True, ensemble_training=False, poe_alpha=1):
        super().__init__()
        self.size_average = size_average
        self.ensemble_training=ensemble_training
        self.poe_alpha = poe_alpha

    def compute_probs(self, inputs):
        prob_dist = F.softmax(inputs, dim=1)
        return prob_dist

    def forward(self, inputs, targets, inputs_adv, second_inputs_adv=None, inputs_adv_is_prob = False):
        targets = targets.view(-1, 1)
        pt = self.compute_probs(inputs)
        pt_adv = inputs_adv if inputs_adv_is_prob else self.compute_probs(inputs_adv)
        if self.ensemble_training:
            pt_adv_second = self.compute_probs(second_inputs_adv)
            joint_pt = F.softmax((torch.log(pt) + torch.log(pt_adv) + torch.log(pt_adv_second)), dim=1)
        else:
            joint_pt = F.softmax((torch.log(pt) + self.poe_alpha*torch.log(pt_adv)), dim=1)
        joint_p = joint_pt.gather(1, targets)
        batch_loss = -torch.log(joint_p)
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

