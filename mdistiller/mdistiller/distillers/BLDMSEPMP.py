import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## THIS HAS NOT BEEN IMPLEMENTED>. DO WHEN YOU ARENT SICK B**CH


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
   
    return cosine_dist.mean()

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean')
    return mse_loss
    

class BLDMSEP(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(BLDMSEP, self).__init__(student, teacher)
        
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_mse = mse(logits_student,logits_teacher)
        
        loss_bld = batch_differences_cosine_distance(logits_student,logits_teacher)
         
        loss_prod = (1 + loss_mse) * (1 + loss_bld)
            
        losses_dict = {
            "loss_prod": loss_prod,
        }
        return logits_student, losses_dict
