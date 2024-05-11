import argparse
import torch
import torch.backends.cudnn as cudnn
import sys
cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict, tiny_imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate, validate_debug
from mdistiller.engine.cfg import CFG as cfg

def print_first_two_layer_weights(model, label=""):
    sys.stdout.write(f"\n{label}")
    first_two_layers = list(model.named_children())[:2]
    for name, layer in first_two_layers:
        for param_name, param in layer.named_parameters():
            sys.stdout.write(f"{name}.{param_name}: {param.data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet", "tiny_imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
            
            print_first_two_layer_weights(model, label="pretrain and print") #this triggered.  
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            print_first_two_layer_weights(model, label="false and print")
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
            print_first_two_layer_weights(model, label="loaded and print")
    elif args.dataset in ("cifar100", "tiny_imagenet"):
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model_dict = tiny_imagenet_model_dict if args.dataset == "tiny_imagenet" else cifar_model_dict
        model, pretrain_model_path = model_dict[args.model]
        model = model(num_classes=num_classes)

        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])

    model = Vanilla(model)
    model = model.cuda()
    model.eval() 
    model = torch.nn.DataParallel(model)
    test_acc, test_acc_top5, test_loss = validate_debug(val_loader, model,"/projectnb/textconv/distill/scripts/eval/o_debug_U.csv")
