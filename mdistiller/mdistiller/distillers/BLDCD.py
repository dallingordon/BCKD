import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
   
    return cosine_dist.mean()

def cosine_dist(logits_student, logits_teacher):
    cosine_distance = 1 - F.cosine_similarity(logits_student, logits_teacher)
    return cosine_distance.mean()
    

class BLDCD(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(BLDCD, self).__init__(student, teacher)
        self.cd_weight = cfg.BLDCD.CD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_cd = self.cd_weight * cosine_dist(logits_student,logits_teacher)
        
        loss_bld = batch_differences_cosine_distance(logits_student,logits_teacher)
            
        losses_dict = {
            "loss_bld": loss_bld,
            "loss_cd": loss_cd,
            
        }
        return logits_student, losses_dict
