import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def batch_differences_cosine_distance(logits_student, logits_teacher):
    batch_size = logits_teacher.shape[0]
    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    return cosine_dist.sum() - batch_size #this gets rid of the ones on diag. 
#later i will add mse.  there is also batch intrinsic mse, bla bla.

class BD(Distiller):
    """Shuffled Label Difference but now its all of em! dg :)"""

    def __init__(self, student, teacher, cfg):
        super(BD, self).__init__(student, teacher)
        

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_bd = batch_differences_cosine_distance(logits_student,logits_teacher)
        
        losses_dict = {
            "loss_bd": loss_bd,
        }
        return logits_student, losses_dict
