from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, GRAD_ACC
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "grad_acc": GRAD_ACC,
}
