import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kds_loss(logits_student, logits_teacher, temperature):
    
    batch_size = logits_teacher.shape[0]
    shuffled_labels_teacher = logits_teacher[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    shuffled_logits_student = logits_student[[batch_size - 1] + [i for i in range(batch_size - 1)]]
    
    log_pred_student = F.log_softmax((logits_student - shuffled_logits_student) / temperature, dim=1)
    pred_teacher = F.softmax((logits_teacher - shuffled_labels_teacher) / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    
    return loss_kd
    #just intrinsic right now.  


class KDS(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KDS, self).__init__(student, teacher)
        self.temperature = cfg.KDS.TEMPERATURE
        self.ce_loss_weight = cfg.KDS.LOSS.CE_WEIGHT
        self.kds_loss_weight = cfg.KDS.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kds = self.kds_loss_weight * kds_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kds": loss_kds,
        }
        return logits_student, losses_dict
