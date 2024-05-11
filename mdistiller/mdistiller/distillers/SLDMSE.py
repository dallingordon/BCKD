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
    return cosine_dist.sum()

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='sum') #changed this to sum.  
    return mse_loss
    

class SLDMSE(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(SLDMSE, self).__init__(student, teacher)
        
        self.mse_weight = cfg.SLDMSE.MSE_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_mse = self.mse_weight * mse(logits_student,logits_teacher)
        
        loss_sld = sldintrinsic_loss(logits_student,logits_teacher)
            
        
        losses_dict = {
            "loss_mse": loss_mse,
            "loss_sld": loss_sld,
        }
        return logits_student, losses_dict
