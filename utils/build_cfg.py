from dassl.config import get_cfg_default


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.MLCCLIP = CN()
    cfg.MLCCLIP.POSITIVE_PROMPT = ""
    cfg.MLCCLIP.NEGATIVE_PROMPT = ""
    cfg.MLCCLIP.FLOAT = False
    cfg.TRAINER.COOP_MLC = CN()
    cfg.TRAINER.COOP_MLC.N_CTX_POS = 16
    cfg.TRAINER.COOP_MLC.N_CTX_NEG = 16
    cfg.TRAINER.COOP_MLC.N_CTX_EVI = 16
    cfg.TRAINER.COOP_MLC.CSC = False
    cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = ""
    cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT = ""
    cfg.TRAINER.COOP_MLC.EVIDENTIAL_PROMPT_INIT = ""
    cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG = 2
    cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS = 1
    cfg.TRAINER.COOP_MLC.ASL_NEG_RATIO = 0.2
    cfg.TRAINER.COOP_MLC.RANK_RATIO = 10
    cfg.TRAINER.COOP_MLC.TEXT_LOGITS_RATIO = 0
    cfg.TRAINER.COOP_MLC.NEG_TEXT_LOGITS_RATIO = 0
    cfg.TRAINER.COOP_MLC.LAMBDA_REG = 0.05
    cfg.TRAINER.COOP_MLC.LOCAL_TOP_K = 18
    cfg.TRAINER.COOP_MLC.TEXT_LOCAL_TOP_K = 10
    cfg.TRAINER.COOP_MLC.ASL_POS_RATIO = 0.0

    cfg.TRAINER.RESNET_IMAGENET = CN()
    cfg.TRAINER.RESNET_IMAGENET.DEPTH = 50
    cfg.TRAINER.FINETUNE = False
    cfg.TRAINER.FINETUNE_BACKBONE = False
    cfg.TRAINER.FINETUNE_ATTN = False

    cfg.DATASET.VAL_SPLIT = ""
    cfg.DATASET.VAL_GZSL_SPLIT = ""
    cfg.DATASET.TEST_SPLIT = ""
    cfg.DATASET.TEST_GZSL_SPLIT = ""
    cfg.DATASET.TRAIN_SPLIT = ""
    cfg.DATASET.ZS_TRAIN = ""
    cfg.DATASET.ZS_TEST = ""
    cfg.DATASET.ZS_TEST_UNSEEN = ""
    cfg.DATALOADER.TRAIN_X.SHUFFLE = True
    cfg.DATALOADER.TRAIN_X.PORTION = 1.
    cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION = 1.
    cfg.DATALOADER.TEST.SHUFFLE = False
    cfg.DATALOADER.VAL = CN()
    cfg.DATALOADER.VAL.SHUFFLE = False
    cfg.DATALOADER.VAL.BATCH_SIZE = 50


    cfg.INPUT.TRAIN = CN()
    cfg.INPUT.TRAIN.SIZE = (224, 224)
    cfg.INPUT.TEST = CN()
    cfg.INPUT.TEST.SIZE = (224, 224)
    cfg.DATASET.MASK_FILE = None
    cfg.OPTIM.BACKBONE_LR_MULT = cfg.OPTIM.BASE_LR_MULT
    cfg.OPTIM.ATTN_LR_MULT = cfg.OPTIM.BASE_LR_MULT
    
    cfg.GAMMA = 1


def reset_cfg(cfg, args):
    if args.positive_prompt:
        cfg.MLCCLIP.POSITIVE_PROMPT = args.positive_prompt
        cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = args.positive_prompt

    if args.negative_prompt:
        cfg.MLCCLIP.NEGATIVE_PROMPT = args.negative_prompt
        cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT = args.negative_prompt

    if args.datadir:
        cfg.DATASET.ROOT = args.datadir

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.print_freq:
        cfg.TRAIN.PRINT_FREQ = args.print_freq

    if args.input_size:
        cfg.INPUT.SIZE = (args.input_size, args.input_size)
        cfg.INPUT.TRAIN.SIZE = (args.input_size, args.input_size)
        cfg.INPUT.TEST.SIZE = (args.input_size, args.input_size)

    if args.train_input_size:
        cfg.INPUT.TRAIN.SIZE = (args.train_input_size, args.train_input_size)
        cfg.INPUT.SIZE = (args.train_input_size, args.train_input_size)

    if args.test_input_size:
        cfg.INPUT.TEST.SIZE = (args.test_input_size, args.test_input_size)

    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.csc:
        cfg.TRAINER.COOP_MLC.CSC = args.csc

    if args.n_ctx_pos:
        cfg.TRAINER.COOP_MLC.N_CTX_POS = args.n_ctx_pos

    if args.n_ctx_neg:
        cfg.TRAINER.COOP_MLC.N_CTX_NEG = args.n_ctx_neg
        
    if args.n_ctx_evi:
        cfg.TRAINER.COOP_MLC.N_CTX_EVI = args.n_ctx_evi
        
    if args.n_ctx_sub:
        cfg.TRAINER.COOP_MLC.N_CTX_SUB = args.n_ctx_sub

    if args.logit_scale:
        cfg.TRAINER.COOP_MLC.LS = args.logit_scale

    if args.gamma_neg:
        cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG = args.gamma_neg

    if args.gamma_pos:
        cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS = args.gamma_pos

    if args.train_batch_size:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size

    if args.finetune:
        cfg.TRAINER.FINETUNE = args.finetune

    if args.finetune_backbone:
        cfg.TRAINER.FINETUNE_BACKBONE = args.finetune_backbone

    if args.finetune_attn:
        cfg.TRAINER.FINETUNE_ATTN = args.finetune_attn

    if args.finetune_text:
        cfg.TRAINER.FINETUNE_TEXT = args.finetune_text

    if args.base_lr_mult:
        cfg.OPTIM.BASE_LR_MULT = args.base_lr_mult

    if args.backbone_lr_mult:
        cfg.OPTIM.BACKBONE_LR_MULT = args.backbone_lr_mult

    if args.text_lr_mult:
        cfg.OPTIM.TEXT_LR_MULT = args.text_lr_mult

    if args.attn_lr_mult:
        cfg.OPTIM.ATTN_LR_MULT = args.attn_lr_mult

    if args.max_epochs:
        cfg.OPTIM.MAX_EPOCH = args.max_epochs

    if args.portion:
        cfg.DATALOADER.TRAIN_X.PORTION = args.portion

    if args.warmup_epochs is not None:
        cfg.OPTIM.WARMUP_EPOCH = args.warmup_epochs

    if args.partial_portion:
        cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION = args.partial_portion

    if args.mask_file:
        cfg.DATASET.MASK_FILE = args.mask_file
        
    if args.gamma:
        cfg.GAMMA = args.gamma
        
    if args.rank_ratio:
        cfg.TRAINER.COOP_MLC.RANK_RATIO = args.rank_ratio
    
    if args.text_logits_ratio is not None:
        cfg.TRAINER.COOP_MLC.TEXT_LOGITS_RATIO = args.text_logits_ratio
        
    if args.neg_text_logits_ratio is not None:
        cfg.TRAINER.COOP_MLC.NEG_TEXT_LOGITS_RATIO = args.neg_text_logits_ratio
        
    if args.lambda_reg is not None:
        cfg.TRAINER.COOP_MLC.LAMBDA_REG = args.lambda_reg
        
    if args.asl_neg_ratio is not None:
        cfg.TRAINER.COOP_MLC.ASL_NEG_RATIO = args.asl_neg_ratio
        
    if args.local_top_k is not None:
        cfg.TRAINER.COOP_MLC.LOCAL_TOP_K = args.local_top_k
        
    if args.text_local_top_k is not None:
        cfg.TRAINER.COOP_MLC.TEXT_LOCAL_TOP_K = args.text_local_top_k
        
    if args.asl_pos_ratio is not None:
        cfg.TRAINER.COOP_MLC.ASL_POS_RATIO = args.asl_pos_ratio
        
        
    if args.weight_decay:
        cfg.OPTIM.WEIGHT_DECAY = args.weight_decay
        
    if args.seed > -1:
        cfg.SEED = args.seed


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg