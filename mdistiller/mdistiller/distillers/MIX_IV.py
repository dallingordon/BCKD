import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Batch Label Differences Cosine Distance Pre-Mean Product
## i should never have kids.  This name would get bullied


def batch_differences_cosine_distance_and_2_products(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    batch_cosine_dist = 2 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    eye = torch.eye(logits_student.shape[0]).to(batch_cosine_dist.device)
    cosine_dist = 2 - F.cosine_similarity(logits_student, logits_teacher)
    outer_cd = torch.outer(cosine_dist,cosine_dist)*(1-eye)
    prod = outer_cd * batch_cosine_dist
    
    return prod.mean()

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean') 
    return mse_loss

def cosine_dist(logits_student, logits_teacher):
    cosine_distance = 1 - F.cosine_similarity(logits_student, logits_teacher)
    return cosine_distance.mean()
    

class MIX_IV(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MIX_IV, self).__init__(student, teacher)
        self.mse_weight = cfg.MIX_IV.MSE_WEIGHT
        self.cd_weight = cfg.MIX_IV.CD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        loss_bld = batch_differences_cosine_distance_and_2_products(logits_student,logits_teacher)
        loss_mse = self.mse_weight * mse(logits_student, logits_teacher)
        loss_cd = self.cd_weight * cosine_dist(logits_student, logits_teacher)

        losses_dict = {
            "loss_bld": loss_bld,
            "loss_mse": loss_mse,
            "loss_cd": loss_cd,
            
            
        }
        return logits_student, losses_dict
