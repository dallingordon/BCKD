import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    ######this other axis may be of interest? i could try that too?
    return cosine_dist.mean()

def cos_both_dim(logits_student, logits_teacher,second_weight):
    cos_1 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-1)
    cos_2 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-2)
    #second_weight weights the second dim weight.
    return cos_1.mean() + second_weight * cos_2.mean()  

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean') 
    return mse_loss

class MIX_V(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MIX_V, self).__init__(student, teacher)
        
        self.bd_weight = cfg.MIX_V.BD_WEIGHT
        self.macd_weight = cfg.MIX_V.MACD_WEIGHT
        self.mse_weight = cfg.MIX_V.MSE_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_bd = self.bd_weight * batch_differences_cosine_distance(logits_student,logits_teacher)
        loss_macdm = cos_both_dim(logits_student,logits_teacher,self.macd_weight)
        loss_mse = self.mse_weight * mse(logits_student, logits_teacher)
            
        
        losses_dict = {
            "loss_bd": loss_bd,
            "loss_macdm": loss_macdm,
            "loss_mse": loss_mse,
        }
        return logits_student, losses_dict
