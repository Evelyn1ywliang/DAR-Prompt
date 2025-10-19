import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')
    parser.add_argument('--prefix', default='', type=str, help='model prefix')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                        help='path to pretrained checkpoint')
    parser.add_argument('--auto_resume', action='store_true', help='if the log folder includes a checkpoint, automatically resume')

    # data-related
    parser.add_argument('--datadir', type=str,  metavar='DIR', help='path to dataset file list')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--train_input_size', type=int, metavar='N', help='input image size')
    parser.add_argument('--num_train_cls', type=int, default=100, help='input image size')
    parser.add_argument('--test_input_size', type=int, metavar='N', help='input image size')
    parser.add_argument('--thre', default=0.5, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('--single_prompt', default='pos', type=str, help='type of single prompt')


    # logging
    parser.add_argument('--output_dir', default='', type=str, help='log path')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency to print the log during the training')
    parser.add_argument('--val_freq_in_epoch', type=int, default=-1,
                        help='frequency to validate the model during the training')

    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # cfg file
    parser.add_argument('--config_file', dest='config_file', type=str, help='network config file path')
    parser.add_argument('--dataset_config_file', dest='dataset_config_file', type=str, help='network config file path')

    # positive prompt & negative prompt
    parser.add_argument('--positive_prompt', dest="positive_prompt", type=str, help='the initial positive prompt for mlc_clip')
    parser.add_argument('--negative_prompt', dest="negative_prompt", type=str, help='the initial negative prompt for mlc_clip')
    # positive prompt & negative prompt number
    parser.add_argument('--n_ctx_pos', dest="n_ctx_pos", type=int, help='the positive prompt for mlc coop')
    parser.add_argument('--n_ctx_neg', dest="n_ctx_neg", type=int, help='the negative prompt for mlc coop')
    parser.add_argument('--n_ctx_evi', dest="n_ctx_evi", type=int, help='the evidential prompt for mlc coop')
    parser.add_argument('--n_ctx_sub', dest="n_ctx_sub", type=int, help='the subordinal prompt for mlc coop')

    parser.add_argument('--lr', dest="lr", type=float, help='the learning rate')
    parser.add_argument('--loss_w', dest="loss_w", type=float, default=1., help='the loss weights')
    parser.add_argument('--bce_ratio', dest="bce_ratio", type=float, default=0., help='the bce loss ratio')

    parser.add_argument('--csc', dest='csc', action='store_true',
                        help='specify the csc')

    parser.add_argument('--logit_scale', dest="logit_scale", type=float, default=100.,
                        help='the logit scale for clip logits')

    parser.add_argument('--gamma_neg', dest="gamma_neg", type=float, default=2.,
                        help='the gamma neg for asymmetric loss')

    parser.add_argument('--gamma_pos', dest="gamma_pos", type=float, default=1.,
                        help='the gamma pos for asymmetric loss')

    parser.add_argument('-p', '--portion', dest="portion", type=float, default=1.,
                        help='the portion of training split used for training')

    parser.add_argument('-pp', '--partial_portion', dest="partial_portion", type=float, default=1+1e-6,
                        help='the portion of partial labels used for training')

    parser.add_argument('--mask_file', dest="mask_file", type=str,
                        help='the mask label for partial labeling')

    parser.add_argument('--train_batch_size', dest="train_batch_size", type=int,
                        help='the batch size for training')

    parser.add_argument('--stop_epochs', dest="stop_epochs", type=int,
                        help='the stop epochs')

    parser.add_argument('--max_epochs', dest="max_epochs", type=int,
                        help='the max epochs')

    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='specify if finetuning the backbone')

    parser.add_argument('--finetune_backbone', dest='finetune_backbone', action='store_true',
                        help='specify if finetuning the backbone')

    parser.add_argument('--finetune_attn', dest='finetune_attn', action='store_true',
                        help='specify if finetuning the backbone')

    parser.add_argument('--finetune_text', dest='finetune_text', action='store_true',
                        help='specify if finetuning the text')

    parser.add_argument('--base_lr_mult', dest='base_lr_mult',  type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--backbone_lr_mult', dest='backbone_lr_mult', type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--text_lr_mult', dest='text_lr_mult', type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--attn_lr_mult', dest='attn_lr_mult', type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--val_every_n_epochs', dest='val_every_n_epochs', type=int, default=1,
                        help='specify if finetuning the backbone')

    parser.add_argument('--warmup_epochs', dest='warmup_epochs', type=int, default=1,
                        help='warm up epochs')

    parser.add_argument('--top_k', dest="top_k", default=3, type=int, help='the top k in validation zsl')
    
    parser.add_argument('--gamma', dest="gamma", default=1., type=float, help='the WTA hyperparameter')
    
    parser.add_argument('--rank_ratio', dest="rank_ratio", default=10., type=float, help='the ratio of text ranking loss')
    
    parser.add_argument('--text_logits_ratio', dest="text_logits_ratio", default=0., type=float, help='the ratio of text logits in the pos output')
    
    parser.add_argument('--neg_text_logits_ratio', dest="neg_text_logits_ratio", default=0., type=float, help='the ratio of text logits in the neg output')
    
    parser.add_argument('--pred_text_ratio', dest="pred_text_ratio", default=1., type=float, help='the ratio of text logits in the softmax output')
    
    parser.add_argument('--lambda_reg', dest="lambda_reg", default=0.05, type=float, help='regularization parameters of grad')
    
    parser.add_argument('--asl_neg_ratio', dest="asl_neg_ratio", default=0.2, type=float, help='the ratio of sentence logis in ASL neg')
    
    parser.add_argument('--local_top_k', dest="local_top_k", default=18, type=int, help='the top k patch in local score')
    
    parser.add_argument('--text_local_top_k', dest="text_local_top_k", default=10, type=int, help='the top k patch in text local score')
    
    parser.add_argument('--asl_pos_ratio', dest="asl_pos_ratio", default=0.0, type=float, help='the ratio of sentence logis in ASL pos')
    
    parser.add_argument('--weight_decay', dest='weight_decay',  type=float, default=0.0005,
                        help='specify if finetuning the backbone')
    
    parser.add_argument('--text_epoch', dest="text_epoch", default=1, type=int, help='text epoch')
    
    parser.add_argument('--bias_epoch', dest="bias_epoch", default=1, type=int, help='bias epoch')
    
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    
    parser.add_argument('--gpu_id', type=int, help='GPU device id used')

    return parser