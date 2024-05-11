import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 2 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
   
    return cosine_dist.mean()

def cosine_dist(logits_student, logits_teacher):
    cosine_distance = 2 - F.cosine_similarity(logits_student, logits_teacher)
    return cosine_distance.mean()
    

class BLDCDP(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(BLDCDP, self).__init__(student, teacher)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_cd = cosine_dist(logits_student,logits_teacher)
        
        loss_bld = batch_differences_cosine_distance(logits_student,logits_teacher)
        
        loss_prod = loss_cd * loss_bld
        losses_dict = {
            "loss_prod": loss_prod,
            
        }
        return logits_student, losses_dict
