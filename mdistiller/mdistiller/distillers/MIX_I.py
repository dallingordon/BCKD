import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def sldintrinsic_loss(logits_student, logits_teacher):
    batch_size = logits_teacher.shape[0]
    shuffled_logits_teacher = logits_teacher[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    shuffled_logits_student = logits_student[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    cosine_dist = 1 - F.cosine_similarity(logits_student - shuffled_logits_student, logits_teacher - shuffled_logits_teacher )
    return cosine_dist.mean() #i made this mean is all.  only dif from sldmse.

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean') 
    return mse_loss

def cos_both_dim(logits_student, logits_teacher):
    cos_1 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-1)
    cos_2 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-2)
    
    return cos_1.mean() + cos_2.mean()   

class MIX_I(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MIX_I, self).__init__(student, teacher)
        
        self.mse_weight = cfg.MIX_I.MSE_WEIGHT
        self.macd_weight = cfg.MIX_I.MACD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_mse = self.mse_weight * mse(logits_student,logits_teacher)
        loss_macdm = self.macd_weight * cos_both_dim(logits_student,logits_teacher)
        loss_sld = sldintrinsic_loss(logits_student,logits_teacher)
            
        
        losses_dict = {
            "loss_mse": loss_mse,
            "loss_sld": loss_sld,
            "loss_macdm": loss_macdm,
        }
        return logits_student, losses_dict
