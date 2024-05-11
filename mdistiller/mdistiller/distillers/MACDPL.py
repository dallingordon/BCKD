import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Multi Axis Cosine Distance Product


def cos_both_dim_prod(logits_student, logits_teacher):
    cos_1 = 2 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-1)
    cos_2 = 2 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-2)
    product = torch.outer(cos_1,cos_2)
    return product.mean()
    

class MACDPL(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MACDPL, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MACDPL.CE_WEIGHT
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        loss_macdp = cos_both_dim_prod(logits_student,logits_teacher)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        losses_dict = {
            "loss_macdp": loss_macdp,
            "loss_ce": loss_ce,
        }
        return logits_student, losses_dict
