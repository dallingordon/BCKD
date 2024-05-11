import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def FLP(x, j = 0.1, k = 1): #(flatter for Lots of Products)
    #converts continuous range between 0 and infinity into a range between 1 and 1 + j
    #this fascilitates doing lots of multiplying so the value doesn't explode.
    return 1 + j * (1 - torch.exp(-k * x))

def batch_contrastive_products(logits_student, logits_teacher, j = 0.1, k = 0.1):
    batch_size = logits_teacher.shape[0]
    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 2 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 ) #1 means identical 3 is opposite
    eye = torch.eye(batch_size).to(cosine_dist.device)
    
    cosine_dist = cosine_dist - eye #gets rid of the 2 along the diag
    
    #this will collapse along the batch dim with products before mean.
    squashed_loss = FLP( cosine_dist, j, k)
    prod_red_loss = torch.prod(squashed_loss, dim = 0)
    return prod_red_loss.mean()
    
    
def mse(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean') #changed this to sum.  
    return mse_loss

class BCP(Distiller):
    """Batch Contrastive Products. Don't Rob the Apostles"""

    def __init__(self, student, teacher, cfg):
        super(BCP, self).__init__(student, teacher)
        self.J = cfg.BCP.J 
        self.K = cfg.BCP.K
        self.mse_weight = cfg.BCP.MSE_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # loss
        loss_mse = self.mse_weight * mse(logits_student,logits_teacher)
        
        loss_bcp = batch_contrastive_products(logits_student
                                              , logits_teacher
                                              , j = self.J
                                              , k = self.K
                                             )
        
        losses_dict = {
            "loss_bcp": loss_bcp,
            "loss_mse": loss_mse,
        }
        return logits_student, losses_dict
