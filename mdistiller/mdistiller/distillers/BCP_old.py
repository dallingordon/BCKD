import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def FLP(x, j = 0.1, k = 1): #(flatter for Lots of Products)
    #converts continuous range between 0 and infinity into a range between 1 and 1 + j
    #this fascilitates doing lots of multiplying so the value doesn't explode.
    return 1 + j * (1 - torch.exp(-k * x))

def batch_contrastive_products(logits_student, logits_teacher,option = 0, j = 0.1, k = 0.1, mse_k = 0.1):
    batch_size = logits_teacher.shape[0]
    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 2 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 ) #1 means identical 3 is opposite
    eye = torch.eye(batch_size).to(cosine_dist.device)
    
    cosine_dist = cosine_dist - eye #gets rid of the 2 along the diag
    
    mse_loss = F.mse_loss(student_differences, teacher_differences, reduction='none').mean(dim=-1) #1 means exactly the same
    in_range_mse_loss = FLP(mse_loss,j=2, k=mse_k) #try this for now.  this puts mse in the same range as cd above
    agg_loss = cosine_dist * in_range_mse_loss
    
    if option == 0:
        return agg_loss.mean()
    if option == 1:
        #this does an inter batch product to try not to rob patricia to pay pauline
        if batch_size % 2 == 0:
            halves = torch.chunk(agg_loss, 2, dim=0)
            inter_batch_prod = halves[0] * halves[1]
            return inter_batch_prod.mean()
        else:
            return agg_loss.mean()
    if option == 2:
        #this will collapse along the batch dim with products before mean.
        squashed_loss = FLP( agg_loss-1, j, k)
        prod_red_loss = torch.prod(squashed_loss, dim = 0)
        return prod_red_loss.mean()
    
    if option == 3:
        #full product collapse.  no robbing any of the apostles to pay any of the others.  full gradient communism
        squashed_loss = FLP( agg_loss-1, j, k)
        prod_red_loss = torch.prod(squashed_loss)
        
        return prod_red_loss.mean()

class BCP(Distiller):
    """Batch Contrastive Products. Don't Rob the Apostles"""

    def __init__(self, student, teacher, cfg):
        super(BCP, self).__init__(student, teacher)
        self.option = cfg.BCP.OPTION
        self.J = cfg.BCP.J 
        self.K = cfg.BCP.K
        self.MSE_K = cfg.BCP.MSE_K

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # loss
        loss_bcp = batch_contrastive_products(logits_student
                                              , logits_teacher
                                              , option = self.option
                                              , j = self.J
                                              , k = self.K
                                              , mse_k = self.MSE_K)
        
        losses_dict = {
            "loss_bcp": loss_bcp,
        }
        return logits_student, losses_dict
