import sys
sys.path.insert(0, '../')
import os
import torch
import torch.nn as nn
import time
from utils.helper import AverageMeter, mAP
from utils.validations import validate, validate_zsl
from utils.asymmetric_loss import TextGuidedAsymmetricLoss, AsymmetricLoss2,AsymmetricLoss_ori
from torch.cuda.amp import autocast
from utils.losses import ranking_loss, contrastive_loss
import clip
import numpy as np
import torch.nn.functional as F
import math
import copy

def train_coop_text_image(data_loader, val_loaders, model, optim, sched, args, cfg, epoch, cls_id=None, logfile=None):
    batch_time = AverageMeter()
    mAP_batches = AverageMeter()
    losses = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)
    Sig = torch.nn.Sigmoid()

    if cls_id is not None:
        num_train_cls = len(cls_id['train'])
    # # switch to evaluate mode
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.image_encoder.train()
    else:
        model.module.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.module.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.module.image_encoder.train()

    criterion = TextGuidedAsymmetricLoss(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)
    criterion2 = AsymmetricLoss2(0, 0, clip=0.0)  # BCE Loss
    criterion3 = AsymmetricLoss_ori(2, 1, clip=0.05)  # vanilla ASL
    
    criterion4 = ranking_loss

    end = time.time()
    
    deb_logits = None
    for i,   (images, sentences, target) in enumerate(data_loader):
        flag=False
        target = target.max(dim=1)[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        images = images.to(device)
        target = target.to(device)
        # identify empty sentence
        filtered_index = [index for index, sentence in enumerate(sentences) if len(sentence.strip()) > 0]
        
        if len(filtered_index) > 0:
            flag = True
            
        if flag:
            sentence_token = clip.tokenize(np.array(sentences)[filtered_index].tolist())
            sentence_token = sentence_token.to(device)

            with torch.no_grad():
                sentence_embedding = model.token_embedding(sentence_token).type(model.dtype)  
            text_features = model.text_encoder.forward_local(sentence_embedding, sentence_token)
            # normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            text_features = text_features.to(device)  
            text_local_features = text_features  
            text_global_features = text_features[torch.arange(text_features.shape[0]), sentence_token.argmax(dim=-1)] 
        else:
            text_features = None
            text_local_features = None
            text_global_features = None
        
        if cls_id is not None:
            # for nus, select 100 out of 925 seen classes, following DualCoOp
            if num_train_cls > args.num_train_cls:
                batch_cls_id = torch.randperm(num_train_cls).cpu().tolist()[:args.num_train_cls]
                # real selected idx
                batch_cls_id_input = [cls_id['train'][a] for a in batch_cls_id]
            else:
                batch_cls_id_input = cls_id['train']
        else:
            batch_cls_id_input = None

        # load the identified biased classes
        bce_idx = torch.load("bad_idx_nus.pt")
        # find the biased classes idx in selected labels this batch
        batch_bce_idx = [idx_ for idx_ in range(args.num_train_cls) if batch_cls_id_input[idx_] in bce_idx]  
        # Map them back to their actual class indices in the overall training set
        real_special_idx = (torch.tensor(batch_cls_id_input)[batch_bce_idx]).tolist()  

        non_bce_idx = batch_cls_id_input

        with autocast():
            # Forward pass over all classes, compute semantic logits
            output, sem_logits, img_sentence_logits, text_features = model.forward_filter_local_text_image(images, text_global_features, text_local_features, filtered_index, non_bce_idx)
            if len(batch_bce_idx) > 0:
                # only forward for biased classes. debiased regulator and debiased logits
                deb_logits, _ = model.forward_biased_class(images, real_special_idx)
                
        if cls_id is not None:
            target1 = target[:, batch_cls_id_input]
            target2 = target[:, real_special_idx]  
            
        if output.dim() == 3:
            if sem_logits is not None:

                sem_logits = sem_logits.type(output.dtype)
                loss_img_tsal = criterion(output, target1, sem_logits, filtered_index, cfg.TRAINER.COOP_MLC.ASL_NEG_RATIO)
                
                
                all_zero_indices = torch.all(target2 == 0, dim=1)
                if len([~all_zero_indices]) > 0:
                    loss_img_bce = criterion2(deb_logits[~all_zero_indices], target2[~all_zero_indices])
                else:
                    loss_img_bce = 0
                loss_img = loss_img_tsal + args.bce_ratio * loss_img_bce
                loss_img *= args.loss_w
                loss_text = args.rank_ratio * args.loss_w * criterion4(sem_logits, target1, scale_ = 1.0, margin_ = 1)

                loss = loss_img + loss_text  

            else:
                sem_logits = 0
                loss = args.loss_w * criterion(output, target, None, None)
        elif args.single_prompt == 'pos':
            loss = args.loss_w * criterion2(output, target)
        elif args.single_prompt == 'neg':
            loss = args.loss_w * criterion3(output, target)
        else:
            raise ValueError

        # update the network
        optim.zero_grad()

        loss_img.backward(retain_graph=True)
        
        # Contrastive Gradient Regularization
        #########################################################
        neg_gradients = model.prompt_learner.ctx_neg.grad
        pos_gradients = model.prompt_learner.ctx_pos.grad
        evi_gradients = model.prompt_learner.ctx_evi.grad
        
        # Flatten the gradient tensor
        ctx_neg_grad_flat = neg_gradients.view(-1).detach()
        ctx_pos_grad_flat = pos_gradients.view(-1).detach()
        ctx_evi_grad_flat = evi_gradients.view(-1).detach()
        norm_grad_pos = pos_gradients / torch.norm(model.prompt_learner.ctx_pos.grad)
        norm_grad_neg = neg_gradients / torch.norm(model.prompt_learner.ctx_neg.grad)

        # pos_evi_sim = F.cosine_similarity(ctx_pos_grad_flat.unsqueeze(0), ctx_evi_grad_flat.unsqueeze(0))
        pos_neg_sim = F.cosine_similarity(ctx_pos_grad_flat.unsqueeze(0), ctx_neg_grad_flat.unsqueeze(0))
        lambda_reg = cfg.TRAINER.COOP_MLC.LAMBDA_REG

        if pos_neg_sim > 0 and lambda_reg > 0:
            pos_grad_norm = pos_gradients / torch.linalg.norm(pos_gradients) 
            changed_grad = neg_gradients - lambda_reg * torch.dot(
                        neg_gradients.flatten(), pos_grad_norm.flatten()
                    ) * pos_grad_norm

        else:   
            changed_grad = neg_gradients
        #############################################################
        
        model.zero_grad()
        optim.zero_grad()
        loss.backward()
        
        model.prompt_learner.ctx_neg.grad = changed_grad  
        

        optim.step()

        losses.update(loss.item(), images.size(0))
        if output.dim() == 3:
            pred = Softmax(output.detach())[:, 1, :]
            
        else:
            pred = Sig(output.detach())
        mAP_value = mAP(target1.cpu().numpy(), pred.cpu().numpy())
        mAP_batches.update(mAP_value, images.size(0))
        
        batch_time.update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                  'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})'.format(
                i, len(data_loader), batch_time=batch_time,
                losses=losses, mAP_batches=mAP_batches), flush=True)
            

        if args.val_freq_in_epoch != -1 and (i + 1) % args.val_freq_in_epoch == 0:
            if len(val_loaders) == 1:
                p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(val_loaders[0], model, args)
                print('Test: [{}/{}]\t '
                      ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                      .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score), flush=True)
            elif len(val_loaders) == 2:
                p_unseen, r_unseen, f1_unseen, mAP_unseen = validate_zsl(val_loaders[0], model, args,
                                                                         cls_id['val_unseen'], is_unseen=True)
                p_gzsl, r_gzsl, f1_gzsl, mAP_gzsl = validate_zsl(val_loaders[1], model, args, cls_id['val_gzsi'])
                
                print('Test: [{}/{}]\t '
                      ' P_unseen {:.2f} \t R_unseen {:.2f} \t F1_unseen {:.2f} \t mAP_unseen {:.2f}\t'
                      ' P_gzsl {:.2f} \t R_gzsl {:.2f} \t F1_gzsl {:.2f} \t mAP_gzsl {:.2f}\t'
                      .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_unseen, r_unseen, f1_unseen, mAP_unseen, p_gzsl, r_gzsl,
                              f1_gzsl, mAP_gzsl), flush=True)
            else:
                raise ValueError

    sched.step()

    return batch_time, losses, mAP_batches

    
def get_prompts(new_model, model, non_ema_idx, ema_idx):
    
    with torch.no_grad():
        non_prompts, non_tokenized_prompts = model.prompt_learner(non_ema_idx)
        ema_prompts, ema_tokenized_prompts = new_model.prompt_learner(ema_idx)
    # neg_ori_prompts, pos_ori_prompts, evi_ori_prompts = torch.chunk(non_prompts, 3, dim=0)
    
    return (non_prompts, non_tokenized_prompts, ema_prompts, ema_tokenized_prompts)