import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Multi-Axis Cosine Distance (mean)


def cos_both_dim(logits_student, logits_teacher):
    cos_1 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-1)
    cos_2 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-2)
    
    return cos_1.mean() + cos_2.mean()

def axes_interactions_loss(logits_student, logits_teacher, axis = 1, cd_collapse_axis = 0):
    #do 1 for batches (bxbxc), 2 for classes (bxcxc)
    student_expanded_a = logits_student.unsqueeze(axis)
    student_expanded_b = logits_student.unsqueeze(axis - 1)
    student_expanded = student_expanded_a * student_expanded_b
    
    teacher_expanded_a = logits_teacher.unsqueeze(axis)
    teacher_expanded_b = logits_teacher.unsqueeze(axis - 1)
    teacher_expanded = teacher_expanded_a * teacher_expanded_b
    
    cd = 1 - torch.nn.functional.cosine_similarity(student_expanded, teacher_expanded, dim=cd_collapse_axis)

    return cd.mean()

class MIX_III(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MIX_III, self).__init__(student, teacher)
        
        self.batch_red_weight = cfg.MIX_III.B_WEIGHT
        self.batch_red_collapse_axis = cfg.MIX_III.B_AXIS
        self.class_red_weight = cfg.MIX_III.C_WEIGHT
        self.class_red_collapse_axis = cfg.MIX_III.C_AXIS


    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        loss_macdm = cos_both_dim(logits_student,logits_teacher)
        loss_b = self.batch_red_weight * axes_interactions_loss(logits_student
                                                                , logits_teacher
                                                                , axis = 1
                                                                , cd_collapse_axis = self.batch_red_collapse_axis)
        loss_c = self.class_red_weight * axes_interactions_loss(logits_student
                                                                , logits_teacher
                                                                , axis = 2
                                                                , cd_collapse_axis = self.class_red_collapse_axis)

        losses_dict = {
            "loss_macdm": loss_macdm,
            "loss_b": loss_b,
            "loss_c": loss_c,
            
        }
        return logits_student, losses_dict
