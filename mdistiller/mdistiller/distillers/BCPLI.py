import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def FLP(x, j = 1, k = 1, e = 1): #(flatter for Lots of Products)
    return 1 + j * (1 - torch.exp(-k * x**e))

def batch_contrastive_products_loss_interaction(logits_student
                                                , logits_teacher
                                                , option = 0
                                                , collapse_option = 0
                                                , j = 0.1
                                                , k = 0.1
                                                , k_o = 0.1
                                                , j_o = 0.1):
    batch_size = logits_teacher.shape[0]
    student_differences = logits_student.unsqueeze(1) - logits_student.unsqueeze(0)
    teacher_differences = logits_teacher.unsqueeze(1) - logits_teacher.unsqueeze(0)
    cosine_dist = 1 - F.cosine_similarity(student_differences, teacher_differences, dim = -1 ) 
    
    cosine_dist = cosine_dist - torch.eye(batch_size).to(cosine_dist.device) #gets rid of the 2 along the diag
    
    squasehd_product = FLP(cosine_dist,j=j, k=k, e=2)
    collapsed_products = torch.prod(squasehd_product, dim = -1).unsqueeze(1) #1 to (1+j) ^^batchsize
    
    if option == 0:
        """
        uses mse, 
        """
        sample_loss = F.mse_loss(logits_student, logits_teacher, reduction='none') + 1 #1 to infinity
        
    if option == 1:
        sample_loss = 2 - F.cosine_similarity(logits_student, logits_teacher, dim=-1) #1 to 3...
    
    if collapse_option == 0:
        print(sample_loss)
        return (collapsed_products*sample_loss).mean()
    
    if collapse_option == 1:
        pre_squashed = (collapsed_products*sample_loss) - 1 
        squashed_output = FLP(pre_squashed,j=j_o, k=k_o, e = 2)
        return torch.prod(squashed_output)

class BCPLI(Distiller):
    """Batch Contrastive Products with Loss Interactions. Don't Rob the Apostles"""

    def __init__(self, student, teacher, cfg):
        super(BCPLI, self).__init__(student, teacher)
        self.option = cfg.BCPLI.OPTION
        self.collapse_option = cfg.BCPLI.COLLAPSE_OPTION
        self.J = cfg.BCPLI.J 
        self.K = cfg.BCPLI.K
        self.J_O = cfg.BCPLI.J_O
        self.K_O = cfg.BCPLI.K_O

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # loss
        loss_bcpli = batch_contrastive_products_loss_interaction(logits_student
                                              , logits_teacher
                                              , option = self.option
                                              , j = self.J
                                              , k = self.K
                                              , k_o = self.K_O
                                              , j_o = self.J_O)
        
        losses_dict = {
            "loss_bcpli": loss_bcpli,
        }
        return logits_student, losses_dict
