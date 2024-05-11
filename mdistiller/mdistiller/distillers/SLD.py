import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## This started as a copy of KD.  I want to add my shuffled label differences

def kd_loss(logits_student, logits_teacher, temperature):
    #print(logits_student.shape, logits_teacher.shape)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    #print(loss_kd)
    #raise ValueError("stopped")
    return loss_kd

def sldcd_loss(logits_student, logits_teacher):
    batch_size = logits_teacher.shape[0]
    shuffled_labels = logits_teacher[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    cosine_dist = 1 - F.cosine_similarity(logits_student - shuffled_labels, logits_teacher - shuffled_labels )
    return cosine_dist.sum()


class SLD(Distiller):
    """Shuffled Label Difference (Cosine Distance) dg :)"""

    def __init__(self, student, teacher, cfg):
        super(SLD, self).__init__(student, teacher)
        self.temperature = cfg.SLD.TEMPERATURE
        self.ce_loss_weight = cfg.SLD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.SLD.LOSS.KD_WEIGHT
        self.sld_loss_weight = cfg.SLD.LOSS.CD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        loss_sld = self.sld_loss_weight * sldcd_loss(logits_student,logits_teacher)
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_sld": loss_sld,
        }
        return logits_student, losses_dict
