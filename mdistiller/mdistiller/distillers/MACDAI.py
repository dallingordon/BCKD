import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Multi-Axis Cosine Distance All Interactions (Mean)


def all_interactions_cd(logits_student, logits_teacher):
    student_batch_interactions = logits_student.unsqueeze(1) * logits_student.unsqueeze(0)
    student_class_interactions = logits_student.unsqueeze(2) * logits_student.unsqueeze(1)

    student_batch_interactions = student_batch_interactions.unsqueeze(-1)  # Shape becomes [b, b, c, 1]
    student_class_interactions = student_class_interactions.unsqueeze(0)  # Shape becomes [1, b, c, c]

    student_expanded = student_batch_interactions * student_class_interactions
    
    teacher_batch_interactions = logits_teacher.unsqueeze(1) * logits_teacher.unsqueeze(0)
    teacher_class_interactions = logits_teacher.unsqueeze(2) * logits_teacher.unsqueeze(1)

    teacher_batch_interactions = teacher_batch_interactions.unsqueeze(-1)  # Shape becomes [b, b, c, 1]
    teacher_class_interactions = teacher_class_interactions.unsqueeze(0)  # Shape becomes [1, b, c, c]

    teacher_expanded = teacher_batch_interactions * teacher_class_interactions
    
    cd_0 =1 - torch.nn.functional.cosine_similarity(student_expanded, teacher_expanded, dim=0)
    cd_1 =1 - torch.nn.functional.cosine_similarity(student_expanded, teacher_expanded, dim=1)
    cd_2 =1 - torch.nn.functional.cosine_similarity(student_expanded, teacher_expanded, dim=2)
    cd_3 =1 - torch.nn.functional.cosine_similarity(student_expanded, teacher_expanded, dim=3)
    
    return cd_0.mean() + cd_1.mean() + cd_2.mean() + cd_3.mean()
    

class MACDAI(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MACDAI, self).__init__(student, teacher)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        loss_macdai = all_interactions_cd(logits_student,logits_teacher)
        

        losses_dict = {
            "loss_macdai": loss_macdai,
            
        }
        return logits_student, losses_dict
