import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

## Copied sldr and made it return both loses, i have abandoned extrinsic at this point.


def batch_differences_cosine_distance(logits_student, logits_teacher):

    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 )
    ######this other axis may be of interest? i could try that too?
    return cosine_dist.mean()

def cos_both_dim(logits_student, logits_teacher,first_weight = 1.0,second_weight = 0.0):
    #this can be used for either axis now
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

class MIX_VI_MAX(Distiller):
    """Shuffled Label Difference (Cosine Distance) with random mse backprop"""

    def __init__(self, student, teacher, cfg):
        super(MIX_VI_MAX, self).__init__(student, teacher)
        
        self.bd_weight = cfg.MIX_VI_MAX.BD_WEIGHT
        self.cda_weight = cfg.MIX_VI_MAX.CDA_WEIGHT
        self.cdb_weight = cfg.MIX_VI_MAX.CDB_WEIGHT
        self.mse_weight = cfg.MIX_VI_MAX.MSE_WEIGHT
        
        self.batch_red_weight = cfg.MIX_VI_MAX.B_WEIGHT
        self.batch_red_collapse_axis = 2 #just setting this, learned it while testing mix_iii
        self.class_red_weight = cfg.MIX_VI_MAX.C_WEIGHT
        self.class_red_collapse_axis = 0 #just setting this
        self.losses_used = cfg.MIX_VI_MAX.LOSSES_R

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        

        loss_bd = self.bd_weight * batch_differences_cosine_distance(logits_student,logits_teacher)
        loss_cda = cos_both_dim(logits_student,logits_teacher,self.cda_weight,0.0)
        loss_cdb = cos_both_dim(logits_student,logits_teacher,0.0,self.cdb_weight)
        loss_mse = self.mse_weight * mse(logits_student, logits_teacher)
        loss_b = self.batch_red_weight * axes_interactions_loss(logits_student
                                                                , logits_teacher
                                                                , axis = 1
                                                                , cd_collapse_axis = self.batch_red_collapse_axis)
        loss_c = self.class_red_weight * axes_interactions_loss(logits_student
                                                                , logits_teacher
                                                                , axis = 2
                                                                , cd_collapse_axis = self.class_red_collapse_axis)  
        loss_list =[loss_bd,loss_cda,loss_cdb,loss_mse,loss_b,loss_c]
        loss_list = loss_list[:self.losses_used] #cuts off any at the end you don't want.
        max_idx = loss_list.index(max(loss_list))
        loss_r = loss_list[max_idx]
        ##consider torch.where()
        losses_dict = {
            "loss_r": loss_r,
            
        }
        return logits_student, losses_dict
