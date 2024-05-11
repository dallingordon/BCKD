import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Batch Label Differences Cosine Distance Pre-Mean Product
## i should never have kids.  This name would get bullied


def batch_differences_cosine_distance_and_2_products(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    batch_cosine_dist = 2 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    eye = torch.eye(logits_student.shape[0]).to(batch_cosine_dist.device)
    cosine_dist = 2 - F.cosine_similarity(logits_student, logits_teacher)
    outer_cd = torch.outer(cosine_dist,cosine_dist)*(1-eye)
    prod = outer_cd * batch_cosine_dist
    
    return prod.mean()
    

class BLDCD2P(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(BLDCD2P, self).__init__(student, teacher)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        loss_bld = batch_differences_cosine_distance_and_2_products(logits_student,logits_teacher)
        

        losses_dict = {
            "loss_bld": loss_bld,
            
        }
        return logits_student, losses_dict
