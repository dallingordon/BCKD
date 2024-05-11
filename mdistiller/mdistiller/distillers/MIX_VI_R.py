import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import random

def generate_class(num_classes = 1):
    return int(random.random() * num_classes) 

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    ######this other axis may be of interest? i could try that too?
    return cosine_dist.mean()

def cos_both_dim(logits_student, logits_teacher,first_weight = 1.0,second_weight = 0.0):
    #this can be used for either axis now
    #but, it is running both even if i set em to 0.  not ideal.  just fyi
    cos_1 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-1)
    cos_2 = 1 - torch.nn.functional.cosine_similarity(logits_student, logits_teacher, dim=-2)
    #second_weight weights the second dim weight.
    return first_weight*cos_1.mean() + second_weight * cos_2.mean()  

def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean') 
    return mse_loss

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

class MIX_VI_R(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MIX_VI_R, self).__init__(student, teacher)
        
        self.bd_weight = cfg.MIX_VI_R.BD_WEIGHT
        self.cda_weight = cfg.MIX_VI_R.CDA_WEIGHT
        self.cdb_weight = cfg.MIX_VI_R.CDB_WEIGHT
        self.mse_weight = cfg.MIX_VI_R.MSE_WEIGHT
        
        self.batch_red_weight = cfg.MIX_VI_R.B_WEIGHT
        self.batch_red_collapse_axis = 2 #just setting this, learned it while testing mix_iii
        self.class_red_weight = cfg.MIX_VI_R.C_WEIGHT
        self.class_red_collapse_axis = 0 #just setting this
        self.losses_used = cfg.MIX_VI_R.LOSSES_R

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        r = generate_class(self.losses_used) #returns 
        
        if r == 0:
            loss_r = self.bd_weight * batch_differences_cosine_distance(logits_student,logits_teacher)
        elif r == 1:
            loss_r = cos_both_dim(logits_student,logits_teacher,self.cda_weight,0.0)
        elif r == 2:
            loss_r = cos_both_dim(logits_student,logits_teacher,0.0,self.cdb_weight)
        elif r == 3:
            loss_r = self.mse_weight * mse(logits_student, logits_teacher)
        elif r == 4:
            loss_r = self.batch_red_weight * axes_interactions_loss(logits_student
                                                                , logits_teacher
                                                                , axis = 1
                                                                , cd_collapse_axis = self.batch_red_collapse_axis)
        elif r == 5:
            loss_r = self.class_red_weight * axes_interactions_loss(logits_student
                                                                , logits_teacher
                                                                , axis = 2
                                                                , cd_collapse_axis = self.class_red_collapse_axis)  
        
        else:
            loss_r = 0.0
        
        losses_dict = {
            "loss_r": loss_r,
            
        }
        return logits_student, losses_dict
