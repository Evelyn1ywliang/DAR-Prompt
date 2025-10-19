import os
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import build_model
from utils.validations import validate_zsl
from opts import arg_parser
from dataloaders import build_dataset
from utils.build_cfg import setup_cfg
# from ball_projection import *

# from tSNE import tsne_vis

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)

    test_split = cfg.DATASET.TEST_SPLIT
    test_gzsl_split =  cfg.DATASET.TEST_GZSL_SPLIT
    test_gzsi_dataset = build_dataset(cfg, test_gzsl_split, cfg.DATASET.ZS_TEST)
    test_gzsi_cls_id = test_gzsi_dataset.cls_id
    test_unseen_dataset = build_dataset(cfg, test_split, cfg.DATASET.ZS_TEST_UNSEEN)
    test_unseen_cls_id = test_unseen_dataset.cls_id
    test_gzsi_loader = torch.utils.data.DataLoader(test_gzsi_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                                   shuffle=cfg.DATALOADER.TEST.SHUFFLE,
                                                   num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    test_unseen_loader = torch.utils.data.DataLoader(test_unseen_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                                     shuffle=cfg.DATALOADER.TEST.SHUFFLE,
                                                     num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)


    classnames = test_gzsi_dataset.classnames

    model, arch_name = build_model(cfg, args, classnames)

    model.eval()


    if os.path.isdir(args.pretrained): # is a folder
        unseen_ckpt = os.path.join(args.pretrained, "unseen_model_best.pth.tar")
        gzsl_ckpt = os.path.join(args.pretrained, "gzsl_model_best.pth.tar")

        # === unseen model ===
        if os.path.isfile(unseen_ckpt):
            print(f"... loading pretrained weights from {unseen_ckpt} for the best unseen zsl", flush=True)
            checkpoint = torch.load(unseen_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            best_epoch = checkpoint["epoch"]
            p_unseen, r_unseen, f1_unseen, mAP_unseen = validate_zsl(
                test_unseen_loader, model, args, test_unseen_cls_id, is_unseen=True
            )
            print(f"Test: [{best_epoch}/{cfg.OPTIM.MAX_EPOCH}]\t "
                f"p_unseen {p_unseen:.2f}\t r_unseen {r_unseen:.2f}\t "
                f"f_unseen {f1_unseen:.2f}\t mAP_unseen {mAP_unseen:.2f}", flush=True)

        # === gzsl model ===
        if os.path.isfile(gzsl_ckpt):
            print(f"... loading pretrained weights from {gzsl_ckpt} for the best gzsl", flush=True)
            checkpoint = torch.load(gzsl_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            best_epoch = checkpoint["epoch"]
            p_gzsl, r_gzsl, f1_gzsl, mAP_gzsl = validate_zsl(
                test_gzsi_loader, model, args, test_gzsi_cls_id
            )
            print(f"Test: [{best_epoch}/{cfg.OPTIM.MAX_EPOCH}]\t "
                f"p_gzsl {p_gzsl:.2f}\t r_gzsl {r_gzsl:.2f}\t "
                f"f_gzsl {f1_gzsl:.2f}\t mAP_gzsl {mAP_gzsl:.2f}", flush=True)
            
    elif os.path.isfile(args.pretrained) and args.pretrained.endswith(".pth.tar"):
        print(f"... loading pretrained weights from {args.pretrained}", flush=True)
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint.get("epoch", -1)
        p_unseen, r_unseen, f1_unseen, mAP_unseen = validate_zsl(test_unseen_loader, model, args, test_unseen_cls_id, is_unseen=True)
        p_gzsl, r_gzsl, f1_gzsl, mAP_gzsl = validate_zsl(test_gzsi_loader, model, args, test_gzsi_cls_id )
        print('Test: [{}/{}]\t '
            ' P_unseen {:.2f} \t R_unseen {:.2f} \t F1_unseen {:.2f} \t mAP_unseen {:.2f}\t'
            ' P_gzsl {:.2f} \t R_gzsl {:.2f} \t F1_gzsl {:.2f} \t mAP_gzsl {:.2f}\t'
            .format(epoch, cfg.OPTIM.MAX_EPOCH, p_unseen, r_unseen, f1_unseen, mAP_unseen, p_gzsl, r_gzsl, f1_gzsl,
                    mAP_gzsl), flush=True)                           


if __name__ == '__main__':
    main()