import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

##SLD with product with mse, then mean.


def sldintrinsic_loss_prod(logits_student, logits_teacher):
    batch_size = logits_teacher.shape[0]
    shuffled_logits_teacher = logits_teacher[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    shuffled_logits_student = logits_student[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    cosine_dist = 2 - F.cosine_similarity(logits_student - shuffled_logits_student, logits_teacher - shuffled_logits_teacher, dim=-1 )
    mse_loss = 1 + F.mse_loss(logits_student, logits_teacher, reduction='none').mean(dim=-1)
    prod = mse_loss*cosine_dist
    return prod.mean()
    

class SLDMSEPROD(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(SLDMSEPROD, self).__init__(student, teacher)
        

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_mse_prod = sldintrinsic_loss_prod(logits_student,logits_teacher)
            
        
        losses_dict = {
            "loss_mse_prod": loss_mse_prod,

        }
        return logits_student, losses_dict
