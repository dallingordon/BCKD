import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def batch_differences_cosine_distance(logits_student, logits_teacher):
    #batch_size = logits_teacher.shape[0]
    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    return cosine_dist.mean()
#later i will add mse.  there is also batch intrinsic mse, bla bla.
def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean') #changed this to sum.  
    return mse_loss

class BDMSEV(Distiller):
    """Shuffled Label Difference but now its all of em! dg :)"""

    def __init__(self, student, teacher, cfg):
        super(BDMSEV, self).__init__(student, teacher)
        self.mse_weight = cfg.BDMSEV.MSE_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_bd = batch_differences_cosine_distance(logits_student,logits_teacher)
        loss_mse = self.mse_weight * mse(logits_student,logits_teacher)
        losses_dict = {
            "loss_bd": loss_bd,
            "loss_mse": loss_mse,
        }
        return logits_student, losses_dict
