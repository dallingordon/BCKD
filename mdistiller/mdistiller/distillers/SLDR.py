import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## This started as a copy of KD.  I want to add my shuffled label differences



def sldcd_loss(logits_student, logits_teacher):
    batch_size = logits_teacher.shape[0]
    shuffled_labels = logits_teacher[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    cosine_dist = 1 - F.cosine_similarity(logits_student - shuffled_labels, logits_teacher - shuffled_labels )
    return cosine_dist.sum()

def sldintrinsic_loss(logits_student, logits_teacher):
    batch_size = logits_teacher.shape[0]
    shuffled_labels_teacher = logits_teacher[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    shuffled_logits_student = logits_student[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    cosine_dist = 1 - F.cosine_similarity(logits_student - shuffled_logits_student, logits_teacher - shuffled_labels_teacher )
    return cosine_dist.sum()

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean')
    return mse_loss
    

class SLDR(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(SLDR, self).__init__(student, teacher)
        
        self.sld_loss_weight = cfg.SLDR.X_WEIGHT
        self.sld_loss_weight_intrinsic = cfg.SLDR.I_WEIGHT
        self.mse_prob = cfg.SLDR.MSE_PROB
        self.mse_weight = cfg.SLDR.MSE_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        random_number = torch.rand(1).item()
        
        if random_number < self.mse_prob:
            loss_r = self.mse_weight * mse(logits_student,logits_teacher)
        else:
            loss_sld = self.sld_loss_weight * sldcd_loss(logits_student,logits_teacher)
            loss_sld_intrinsic = self.sld_loss_weight_intrinsic * sldintrinsic_loss(logits_student,logits_teacher)
            loss_r = loss_sld + loss_sld_intrinsic
        
        losses_dict = {
            "loss_r": loss_r,
        }
        return logits_student, losses_dict
