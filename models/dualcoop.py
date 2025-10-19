import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['dualcoop', 'DualCoop']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    # model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)
    if 'ViT' in backbone_name:
        model = clip.build_model(state_dict or model.state_dict())
    else:
        model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)
        # model = clip.build_model(state_dict or model.state_dict())


    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward_local(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection

        return x
    
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS
        n_ctx_neg = cfg.TRAINER.COOP_MLC.N_CTX_NEG
        n_ctx_evi = cfg.TRAINER.COOP_MLC.N_CTX_EVI
        n_ctx_sub = cfg.TRAINER.COOP_MLC.N_CTX_SUB
        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT.strip()
        ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT.strip()
        ctx_init_evi = cfg.TRAINER.COOP_MLC.EVIDENTIAL_PROMPT_INIT.strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
                ctx_vectors_evi = torch.empty(n_cls, n_ctx_evi, ctx_dim, dtype=dtype)
                ctx_vectors_sub = torch.empty(n_cls, n_ctx_sub, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
                ctx_vectors_evi = torch.empty(n_ctx_evi, ctx_dim, dtype=dtype)
                ctx_vectors_sub = torch.empty(n_ctx_sub, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            nn.init.normal_(ctx_vectors_evi, std=0.02)
            nn.init.normal_(ctx_vectors_sub, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
            prompt_prefix_evi = " ".join(["X"] * n_ctx_evi)
            prompt_prefix_sub = " ".join(["X"] * n_ctx_sub)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        self.ctx_evi = nn.Parameter(ctx_vectors_evi)  # to be optimized
        self.ctx_sub = nn.Parameter(ctx_vectors_sub)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]
        prompts_evi = [prompt_prefix_evi + " " + name + "." for name in classnames]
        prompts_sub = [prompt_prefix_sub + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        tokenized_prompts_evi = []
        tokenized_prompts_sub = []
        for p_pos, p_neg, p_evi, p_sub in zip(prompts_pos, prompts_neg, prompts_evi, prompts_sub):
        # for p_pos, p_neg, p_evi in zip(prompts_pos, prompts_neg, prompts_evi):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
            tokenized_prompts_evi.append(clip.tokenize(p_evi))
            tokenized_prompts_sub.append(clip.tokenize(p_sub))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        tokenized_prompts_evi = torch.cat(tokenized_prompts_evi)
        tokenized_prompts_sub = torch.cat(tokenized_prompts_sub)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            embedding_evi = clip_model.token_embedding(tokenized_prompts_evi).type(dtype)
            embedding_sub = clip_model.token_embedding(tokenized_prompts_sub).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])
        self.register_buffer("token_prefix_evi", embedding_evi[:, :1, :])
        self.register_buffer("token_suffix_evi", embedding_evi[:, 1 + n_ctx_evi:, :])
        self.register_buffer("token_prefix_sub", embedding_sub[:, :1, :])
        self.register_buffer("token_suffix_sub", embedding_sub[:, 1 + n_ctx_sub:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.n_ctx_evi = n_ctx_evi
        self.n_ctx_sub = n_ctx_sub
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi, tokenized_prompts_sub], dim=0)  # torch.Tensor
        # tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi], dim=0)
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_evi = self.ctx_evi
        

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)  # [100, 42, 512]
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)  # [100, 42, 512]
        else: 
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]
                                
        if ctx_evi.dim() == 2:
            if cls_id is None:
                ctx_evi = ctx_evi.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_evi = ctx_evi.unsqueeze(0).expand(len(cls_id), -1, -1)  # [100, 42, 512]
        else:
            if cls_id is not None:
                ctx_evi = ctx_evi[cls_id]
                

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            prefix_evi = self.token_prefix_evi
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
            suffix_evi = self.token_suffix_evi
        else:
            prefix_pos = self.token_prefix_pos[cls_id]  
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]
            prefix_evi = self.token_prefix_evi[cls_id]
            suffix_evi = self.token_suffix_evi[cls_id]


        prompts_pos = torch.cat(   
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        
        prompts_evi = torch.cat(
            [
                prefix_evi,  # (n_cls, 1, dim)
                ctx_evi,  # (n_cls, n_ctx, dim)
                suffix_evi,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts_pos, prompts_evi], dim=0)  # [300, 77, 512]

        if cls_id is not None:
            tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi, _ = torch.chunk(self.tokenized_prompts,4, dim=0)  # 每个是[1006,77]
            # tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi = torch.chunk(self.tokenized_prompts,3, dim=0)  # 每个是[1006,77]
            tokenized_prompts_neg = tokenized_prompts_neg[cls_id]
            tokenized_prompts_evi = tokenized_prompts_evi[cls_id]
            tokenized_prompts_pos = tokenized_prompts_pos[cls_id]
            
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts


        return prompts, tokenized_prompts
    

    def forward_only_bias(self, cls_id=None): 
        ctx_sub = self.ctx_sub
        ctx_neg = self.ctx_neg
        
        if ctx_sub.dim() == 2:
            if cls_id is None:
                ctx_sub = ctx_sub.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_sub = ctx_sub.unsqueeze(0).expand(len(cls_id), -1, -1)  # [100, 42, 512]
        else:
            if cls_id is not None:
                ctx_sub = ctx_sub[cls_id]
                
        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)  # [100, 42, 512]
        else: 
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]
                
        if cls_id is None:
            prefix_sub = self.token_prefix_sub
            suffix_sub = self.token_suffix_sub
            prefix_neg = self.token_prefix_neg
            suffix_neg = self.token_suffix_neg
        else:
            prefix_sub = self.token_prefix_sub[cls_id]
            suffix_sub = self.token_suffix_sub[cls_id]
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]
            
        prompts_sub = torch.cat(
            [
                prefix_sub,  # (n_cls, 1, dim)
                ctx_sub,  # (n_cls, n_ctx, dim)
                suffix_sub,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        
        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        
        prompts = torch.cat([prompts_neg, prompts_sub], dim=0)
            
        if cls_id is not None:
            tokenized_prompts_neg, _, _, tokenized_prompts_sub = torch.chunk(self.tokenized_prompts,4, dim=0)  
            tokenized_prompts_sub = tokenized_prompts_sub[cls_id]
            tokenized_prompts_neg = tokenized_prompts_neg[cls_id]
            # tokenized_prompts = tokenized_prompts_sub
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_sub], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts
        
        return prompts, tokenized_prompts

class CLIPPrompt(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.classnames = classnames
        # self.template = "a photo of a"
        
    def forward(self, cls_id):
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.classnames[cls_id]])
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {self.classnames[i]}") for i in cls_id])
        # neg_text_inputs = torch.cat([clip.tokenize(f"a photo of not a {self.classnames[i]}") for i in cls_id])
        text_inputs = torch.cat([clip.tokenize(f"there is a {self.classnames[i]} in the scene") for i in cls_id])
        neg_text_inputs = torch.cat([clip.tokenize(f"there is not a {self.classnames[i]} in the scene") for i in cls_id])
        return text_inputs, neg_text_inputs
            

class DualCoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)
        self.clip_prompt = CLIPPrompt(cfg, classnames)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.gamma = cfg.GAMMA
        self.token_embedding = clip_model.token_embedding
        self.sub_id = None
        # self.img_tokens = nn.Parameter(ctx_vectors_pos)  # to be optimized
        # self.init_img_tokens()
        
        temperature_pos = torch.tensor(3, dtype=self.dtype)  # 50
        self.temperature_pos = nn.Parameter(temperature_pos)
        temperature_neg = torch.tensor(3, dtype=self.dtype)  # 50
        self.temperature_neg = nn.Parameter(temperature_neg)
        
        spatial_T = torch.tensor(3.0, dtype=self.dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        
    def init_img_tokens(self):
        ctx_vectors_img = torch.empty(1, 2048, dtype=self.dtype)
        nn.init.normal_(ctx_vectors_img, std=0.02)
        self.img_tokens = nn.Parameter(ctx_vectors_img)

    def forward(self, image, cls_id=None):
        # get image and text features
        image_features, attn_weights, img_feat = self.image_encoder(image.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        # image_features = self.image_encoder(image.type(self.dtype))
        
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        
        """
        for image_text
        """
        n_class = text_features.shape[0] // 3
        text_features_image = text_features[:2*n_class]  
        text_features_st = text_features[2*n_class:]  
        # local global
        image_features_g = image_features[0]/ image_features[0].norm(dim=-1, keepdim=True)
        image_features_l = image_features[1:] / image_features[1:].norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # image_features_g = img_feat[0]
        # image_features_l = img_feat[1:] 
        logit_scale_pos = self.temperature_pos.exp() 
        logit_scale_neg = self.temperature_neg.exp() 
        
        output_g = image_features_g @ text_features_image.t()  # [32, 200]
        output_l = torch.topk(image_features_l.permute(1, 0, 2) @ text_features_image.t(),k=self.cfg.TRAINER.COOP_MLC.LOCAL_TOP_K, dim=1)[0].mean(dim=1)  # [32, 200]
        

        neg_prompt_features, pos_prompt_features = torch.chunk(text_features_image, 2, dim=0)
       
        output_pos = logit_scale_pos * image_features_norm @ pos_prompt_features.t()   
        output_neg = logit_scale_neg * image_features_norm @ neg_prompt_features.t() 
        output_evi = image_features_norm @ text_features_st.t()
        
        map_pos = image_features_norm @ pos_prompt_features.t()
        
        # w_pos = (map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
        w_pos = 1+(map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)

        output_pos *= w_pos
        
        # WTA module following DualCoOp++
        prob_ = torch.nn.functional.softmax(output_evi * self.spatial_T.exp(), dim=0)  
        output_pos = torch.sum(output_pos * prob_, dim=0)
        output_neg = torch.sum(output_neg * prob_, dim=0)

        img_sentence_logits = image_features_g @ text_features_st.t() + \
                torch.topk(image_features_l.permute(1, 0, 2) @ text_features_st.t(),
                           k=self.cfg.TRAINER.COOP_MLC.LOCAL_TOP_K, dim=1)[0].mean(dim=1)


        logits = torch.cat([torch.unsqueeze(output_neg,1), torch.unsqueeze(output_pos,1)], dim=1)

        return logits, img_sentence_logits
    
    
    def forward_filter_local_text_image(self, image, st_global_feature, st_local_feature, choose_index=None, cls_id=None):

        prompts, tokenized_prompts = self.prompt_learner(cls_id)  
        prompts.retain_grad()
            
        text_features = self.text_encoder(prompts, tokenized_prompts)  

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # devide the text
        neg_prompt_features, pos_prompt_features, st_text_feature = torch.chunk(text_features, 3, dim=0)

        image_features, attn_weights, img_feat = self.image_encoder(image.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        image_features_g = image_features[0]/ image_features[0].norm(dim=-1, keepdim=True)
        image_features_l = image_features[1:] / image_features[1:].norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale_pos = self.temperature_pos.exp() 
        logit_scale_neg = self.temperature_neg.exp() 
        
        b, c = text_features.shape

        output_pos = logit_scale_pos * image_features_norm @ pos_prompt_features.t()   
        output_neg = logit_scale_neg * image_features_norm @ neg_prompt_features.t() 
        output_evi = image_features_norm @ st_text_feature.t()
        
        map_pos = image_features_norm @ pos_prompt_features.t()
        
        # w_pos = (map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
        w_pos = 1+(map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
        output_pos *= w_pos
        
        # WTA module following DualCoOp++
        prob_ = torch.nn.functional.softmax(output_evi * self.spatial_T.exp(), dim=0)  
        output_pos = torch.sum(output_pos * prob_, dim=0)
        output_neg = torch.sum(output_neg * prob_, dim=0)
        
        if st_global_feature is not None: 

            st_global = 10 * st_global_feature @ st_text_feature.t()  
            st_local = 10 * torch.topk(st_local_feature @ st_text_feature.t(),
                                       k=self.cfg.TRAINER.COOP_MLC.TEXT_LOCAL_TOP_K, dim=1)[0].mean(dim=1)
            
            sentence_logits = st_global + st_local
            
            img_sentence_logits = image_features_g[choose_index] @ st_text_feature.t() + \
                torch.topk(image_features_l.permute(1, 0, 2)[choose_index] @ st_text_feature.t(),
                           k=self.cfg.TRAINER.COOP_MLC.LOCAL_TOP_K, dim=1)[0].mean(dim=1)
            
        else:
            sentence_logits = None
        
        image_logits = torch.cat([torch.unsqueeze(output_neg,1), torch.unsqueeze(output_pos,1)], dim=1)
        
        return image_logits, sentence_logits, img_sentence_logits, prompts
    
    
    def forward_biased_class(self, images, cls_id):
        
        prompts, tokenized_prompts = self.prompt_learner.forward_only_bias(cls_id)
        
        text_features = self.text_encoder(prompts, tokenized_prompts)  

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        neg_prompt_features, sub_prompt_features = torch.chunk(text_features, 2, dim=0)
        
        image_features, attn_weights, img_feat = self.image_encoder(images.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [49, 32, 1024]

        # local global
        image_features_g = image_features[0]/ image_features[0].norm(dim=-1, keepdim=True)
        image_features_l = image_features[1:] / image_features[1:].norm(dim=-1, keepdim=True)
        
        # neg
        output_image_neg_g = image_features_g @ neg_prompt_features.t()  
        output_image_neg_l = torch.topk(image_features_l.permute(1, 0, 2) @ neg_prompt_features.t(),
                                        k=self.cfg.TRAINER.COOP_MLC.LOCAL_TOP_K , dim=1)[0].mean(dim=1)  # [32, 200]
        output_image_neg = (output_image_neg_g + output_image_neg_l) * 20
        
        # debiased logits
        output_image_sub_g = image_features_g @ sub_prompt_features.t()  
        output_image_sub_l = torch.topk(image_features_l.permute(1, 0, 2) @ sub_prompt_features.t(),
                                        k=self.cfg.TRAINER.COOP_MLC.LOCAL_TOP_K, dim=1)[0].mean(dim=1)  # [32, 200]
        output_image_sub = (output_image_sub_g + output_image_sub_l) * 20
        
        output_image = torch.stack((output_image_neg, output_image_sub), dim=1)

        return output_image, None

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
            
        return params
    
    def only_pos_neg_prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name and 'ctx_evi' not in name:
                params.append(param)
            
        return params
    


def dualcoop(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building dualcoop")
    model = DualCoop(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model
