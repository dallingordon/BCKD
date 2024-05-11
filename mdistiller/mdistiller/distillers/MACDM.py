import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Multi-Axis Cosine Distance (mean)


def cos_both_dim(logits_student, logits_teacher):
    cos_1 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-1)
    cos_2 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-2)
    
    return cos_1.mean() + cos_2.mean()
    

class MACDM(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MACDM, self).__init__(student, teacher)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        loss_macdm = cos_both_dim(logits_student,logits_teacher)
        

        losses_dict = {
            "loss_macdm": loss_macdm,
            
        }
        return logits_student, losses_dict
