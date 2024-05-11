import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
   
    return cosine_dist.mean()

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean')
    return mse_loss
    

class BLDMSE(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(BLDMSE, self).__init__(student, teacher)
        
        self.mse_weight = cfg.BLDMSE.MSE_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_mse = self.mse_weight * mse(logits_student,logits_teacher)
        
        loss_bld = batch_differences_cosine_distance(logits_student,logits_teacher)
            
        losses_dict = {
            "loss_mse": loss_mse,
            "loss_bld": loss_bld,
        }
        return logits_student, losses_dict
