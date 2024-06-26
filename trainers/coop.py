import os.path
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
# from dassl.metrics import evaluate_test
# from dassl.metrics import evaluate_valid
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from SASRec import model as sasr_model
# from clip import clip
from SASRec import sasrec  #init
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from SASRec.model import *
from arguments import *
from typing import Union, List
from collections import OrderedDict


_tokenizer = sasrec.SimpleTokenizer()
args = args_S()

def load_clip_to_cpu(cfg): #下载CLIP模型，将CLIP加载到CPU
    backbone_name = cfg.MODEL.BACKBONE.NAME  # 骨干的名字是传参时，cfg的模型骨干名字
    url = sasrec._MODELS[backbone_name]  # 这里跳到clip.py中去，去得到那个下载链接
    print("The name of model:",url)
    # model_path = clip._download(url)
    # if os.path.exists(url):
    #     model_path = url
    # urls = url.split(',')
    url_E = url #urls[0]
    # url_V= urls[1]

    state_dict_E = torch.load(url_E, map_location="cpu")
    # state_dict_V = state_dict_E#torch.load(url_V, map_location="cpu")

    # try:
        # loading JIT archive  加载JIT存档
        # model = torch.jit.load(url, map_location="cpu").eval()  # 模型加载，并变成eval mode 从'/home/wangchunxiao/.cache/clip/RN50.pt'路径中加载模型
        # state_dict = None
        # state_dict = torch.load(url, map_location="cpu")

    # except RuntimeError:
    #     state_dict = torch.load(url, map_location="cpu")

    model_E = sasr_model.build_model(state_dict_E, cfg)#.to(cfg.TRANSFORMER.DEVICE)  # 模型每层都配置完毕，并且转换为了fp16格式
    # model_V = sasr_model.build_model(state_dict_V, cfg)

    return model_E


class TextEncoder(nn.Module):  # 文本编码
    def __init__(self,cfg):
        super().__init__()
        self.dropout_rate =0 #0.2  # 可以传过来，传过来就是与cop中的dropout一样
        self.hidden_units = cfg.TRANSFORMER.HIDDEN_UNIT
        self.num_heads =1  #
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.attention_layers = torch.nn.ModuleList()#SAS_model.attention_layers
        self.attention_layernorms =torch.nn.ModuleList() # SAS_model.attention_layernorms
        self.forward_layernorms = torch.nn.ModuleList() #SAS_model.forward_layernorms
        self.forward_layers = torch.nn.ModuleList() #SAS_model.forward_layers
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8) # SAS_model.last_layernorm
        self.dev = cfg.TRANSFORMER.DEVICE
        self.num_blocks = 2 #变成2更好一些
        print("num_heads=", self.num_heads, "num_block=", self.num_blocks, "drop_rate_textencoder=", self.dropout_rate)

        # self.maxlen =90
        # self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_units)

        for _ in range(self.num_blocks):  # 循环两遍
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,  # 多头的将总的hidden_unit除成num_heads份
                                                            self.num_heads,
                                                            self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, prompts, tokenized_prompts=None):  # prompt=(4000,77,512) tokenzied_prompt(4000,77)
        x = prompts
        x *= prompts.shape[2] ** 0.5
        # positions = np.tile(np.array(range(prompts.shape[1])), [prompts.shape[0],1])  # (128,200)np.tile(array(range（200）),[128,1])把array(range(200))横向纵向的赋复制为(128,1)
        # ndarray(32,77)
        # x += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # pos_emb(77,50)
        x = self.emb_dropout(x)  # (32,77,512)

        # timeline_mask = torch.cuda.BoolTensor(tokenized_prompts == 0).to(self.dev)  # 时间线掩码，将0的列全部变成Ture，不等于0的变为False(32,77)
        # x *= ~timeline_mask.unsqueeze(-1)  # 在最后一个维度扩充broadcast in last dim(32,77)--(32,77,1),seqs：（32,77，512）--（32，77，512）
        # ~布尔值反转，也就是0=False，不是0的是True。
        tl = x.shape[1]  # tl=77 time dim len for enforce causality强化因果关系的时间维度
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # 生成(77,77)维度的布尔类型
        # （77,77）布尔值下三角矩阵，再翻转值True-False
        for i in range(len(self.attention_layers)):  # range(0,2),这里是多头注意力，只有两层。可以像常规Python列表一样索引，但是它包含的模块已正确注册，所有人都可以看到
            x = torch.transpose(x, 0, 1)  # 在seqs的（0，1）维进行转置
            x = x.float()  # 20220916加
            Q = self.attention_layernorms[i](x)  # 注意力前layernorm，Q是最终的x
            mha_outputs, _ = self.attention_layers[i](Q, x, x) #, attn_mask=attention_mask)  # 注意力网络，多头注意力输出
            x = Q + mha_outputs  # Q+output(Q)：浅绿色x+浅粉 色z
            x = torch.transpose(x, 0, 1)  # （32，77，512）12132,61,50

            x = self.forward_layernorms[i](x)  # 将残差后的数值送入Laynorm，得到整个句子经过多头注意力的特征
            x = self.forward_layers[i](x)  # 前馈神经网络
            # x *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(x)  # (U, T, C) -> (U, -1, C) 输入一个张量，得到(32,77,512)
        # 这里放在每个forward之后。每个句子有一个输入对应一个输出
        # log_feats = log_feats.type(self.seq_text_projection.dtype)  # (32,77,512),seq_text_project=torch.float16
        # seq_features = torch.max(log_feats, 1).values @ self.seq_text_projection  # log_feats:(32,77,512)-> (32,512)*(512,1024)
        # seq_features = torch.max(log_feats, 1).values
        return log_feats
#guo
class SeqEncoder(nn.Module):  # 句子编码
    def __init__(self, SAS_model):
        super().__init__()
        self.pos_emb = SAS_model.pos_emb
        self.emb_dropout = SAS_model.emb_dropout
        self.attention_layers = SAS_model.attention_layers
        self.attention_layernorms = SAS_model.attention_layernorms
        self.forward_layernorms = SAS_model.forward_layernorms
        self.forward_layers = SAS_model.forward_layers
        self.last_layernorm = SAS_model.last_layernorm
        self.dev = SAS_model.dev
        self.item_emb= SAS_model.item_emb

    def forward(self, log_seqs):  # log_seqs(32,77),embeds_seq(32,77,512)
        seqs = self.item_emb(torch.cuda.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0],1])  # (128,200)np.tile(array(range（200）),[128,1])把array(range(200))横向纵向的赋复制为(128,1)
        # ndarray(32,77)  test(1,77)
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # pos_emb(77,50)  (1,77,512)
        seqs = self.emb_dropout(seqs)  # (32,77,512)

        timeline_mask = torch.cuda.BoolTensor(log_seqs == 0).to(self.dev)  # 时间线掩码，将0的列全部变成Ture，不等于0的变为False(32,77)
        seqs *= ~timeline_mask.unsqueeze(-1)  # 在最后一个维度扩充broadcast in last dim(32,77)--(32,77,1),seqs：（32,77，512）--（32，77，512）
        # ~布尔值反转，也就是0=False，不是0的是True。
        tl = seqs.shape[1]  # tl=77 time dim len for enforce causality强化因果关系的时间维度
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # 生成(77,77)维度的布尔类型
        # （77,77）布尔值下三角矩阵，再翻转值True-False
        for i in range(len(self.attention_layers)):  # range(0,2),这里是多头注意力，只有两层。可以像常规Python列表一样索引，但是它包含的模块已正确注册，所有人都可以看到
            seqs = torch.transpose(seqs, 0, 1)  # 在seqs的（0，1）维进行转置
            # seqs =seqs.float()  # 20220916加
            Q = self.attention_layernorms[i](seqs)  # 注意力前layernorm，Q是最终的x
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)  # 注意力网络
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs  # Q+output(Q)：浅绿色x+浅粉色z
            seqs = torch.transpose(seqs, 0, 1)  # （32，77，512）

            seqs = self.forward_layernorms[i](seqs)  # 将残差后的数值送入Laynorm，得到整个句子经过多头注意力的特征
            seqs = self.forward_layers[i](seqs)  # 前馈神经网络
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C) 输入一个张量，得到(32,77,512)
        # 这里放在每个forward之后。每个句子有一个输入对应一个输出
        # log_feats = log_feats.type(self.seq_text_projection.dtype)  # (32,77,512),seq_text_project=torch.float16
        # seq_features = torch.max(log_feats, 1).values @ self.seq_text_projection  # log_feats:(32,77,512)-> (32,512)*(512,1024)
        # seq_features = torch.max(log_feats, 1).values
        return log_feats  # (class,1024)
###Agree#####
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )
    def forward(self, x, mask=None):
        # out = self.linear(x)
        # weight = F.softmax(out.view(1, -1), dim=1)
        # return weight
        out = self.linear(x)
        if mask is not None:  # 这是针对item aggregation中的attention部分
            out = out.masked_fill(mask, -100000)  # 在mask值为1的位置处用value填充-np.inf
            weight = F.softmax(out, dim=1)
            return weight
        else:
            weight = F.softmax(out, dim=1) #F.softmax(out.view(1, -1), dim=1)
        return weight  # 得到attention weight
#定义Transformer,clip
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16.子类torch的LayerForm以处理fp16"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))  # 先ln_1:512;再经过多头注意力，512-512；+x：残差网络
        x = x + self.mlp(self.ln_2(x))  # 先ln_2：512；再mlp:512-2048,2048-512;+x：残差网络
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

#可学习的东西都放到这里
class PromptLearner(nn.Module):
    def __init__(self, cfg, sasr_model_E, num_E, num_V):
        super().__init__()
        # n_cls = cfg.DATALOADER.NUM_CLASS  # len(classnames)  #n_cls代表类别数 20220914
        n_cls_E = num_E   # 3398
        n_cls_V = num_V  # 16462
        n_ctx = cfg.TRAINER.COOP.N_CTX  # 16
        ctx_init = cfg.TRAINER.COOP.CTX_INIT #表示用于初始化的文本
        ctx_dim = sasr_model_E.hidden_units  #E and V are the same


        #################2022 1213
        emb_num = 3#5#20 #30 #25 #20 独特
        emb_num_S = 5#3#None#20 MB=3,FK=1 共享
        self.context_embedding_E = nn.Embedding(emb_num, cfg.TRANSFORMER.HIDDEN_UNIT)
        self.context_embedding_V = nn.Embedding(emb_num, cfg.TRANSFORMER.HIDDEN_UNIT)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, cfg.TRANSFORMER.HIDDEN_UNIT) #share
        self.context_embedding_s_V = self.context_embedding_s_E # nn.Embedding(emb_num_S, cfg.TRANSFORMER.HIDDEN_UNIT)  # share

        #################################################################
        #new guo "?
        drop_out = 0.3 #应该是几？0.2forFK,0.3forMB

        #
        self.attention_E = AttentionLayer(2 * cfg.TRANSFORMER.HIDDEN_UNIT, drop_out)
        self.attention_V = AttentionLayer(2 * cfg.TRANSFORMER.HIDDEN_UNIT, drop_out)
        print("drop_out=",drop_out)
        if emb_num is not None:
            print("emb_dute=",emb_num)
        if emb_num_S is not None:
            print("emb_S=", emb_num_S)
        self.text_encoder_E = TextEncoder(cfg)
        self.text_encoder_V = TextEncoder(cfg)


        #原来是在clip中定义的，应该是已经学好的
        if ctx_init:  # 执行这里，这里可以用于加上新的词embedding

            #########################################初始化
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                if isinstance(m, nn.Embedding):
                    nn.init.xavier_normal_(m.weight)
            #################################################
            embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
            embedding_V = self.context_embedding_V(torch.LongTensor(list(range(emb_num))))
            embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
            embedding_S_V = self.context_embedding_s_V(torch.LongTensor(list(range(emb_num_S))))

            ctx_vectors_E = embedding_E
            ctx_vectors_V = embedding_V
            ctx_vectors_S_E = embedding_S_E
            ctx_vectors_S_V = embedding_S_V


        else: #不执行
            # random initialization 随机初始化
            if cfg.TRAINER.COOP.CSC:  # False
                print("Initializing class-specific contexts")  # 初始化类特定上下文
                ctx_vectors = torch.empty(n_cls_E, n_ctx, ctx_dim)#, dtype=dtype)
            else:
                print("Initializing a generic context")  # 初始化通用上下文
                ctx_vectors = torch.empty(n_ctx, ctx_dim)#, dtype=dtype)  # 生成一个16*512的未知张量（数字随机生成），16是初始字符串的长度，512是词向量维度
            nn.init.normal_(ctx_vectors, std=0.02)  # 使用从法线绘制的值填充输入张量
            prompt_prefix = " ".join(["X"] * n_ctx)  # 词前缀=‘16个X，每个X中间有空格’

        #####所有的item将会共享self.ctx,根据不同的item数量调整不同的前缀和后缀个数
        #a photo of的向量表示,requires_grad变成true
        self.ctx_E = nn.Parameter(ctx_vectors_E)  # 4*50  to be optimized，被所有的item共享 待优化 参数层将ctx_vectors放进去，通过parameter将词向量带入进模型中，将此变量成为模型的一部分，根据训练就可以改动，想让这些参数在学习的过程中达到最优
        self.ctx_V = nn.Parameter(ctx_vectors_V)
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)
        self.ctx_S_V = nn.Parameter(ctx_vectors_S_V)
        with torch.no_grad():  # 不是参数，token_embedding是一样的，根据prompts E取出来
            embedding_class_E = sasr_model_E.item_emb(torch.LongTensor(list(range(n_cls_E)))).unsqueeze(1)  # 不更新的用的预训练的  n_cls_E是加了30之后的类数量
        with torch.no_grad():
            embedding_class_V = sasr_model_E.item_emb(torch.LongTensor(list(range(n_cls_E, n_cls_E+n_cls_V)))).unsqueeze(1)
        self.register_buffer("token_prefix_E", embedding_class_E)  # buffer中的内容不会被更新
        self.register_buffer("token_prefix_V", embedding_class_V)  # SOS词前缀
        # 在内存中定一个不太被重视的模型参数常量，保证模型保存和加载时可以写入和读出
        self.n_cls_E = n_cls_E
        self.n_cls_V = n_cls_V
        self.n_ctx = n_ctx  # 16
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.transformer_width = cfg.TRANSFORMER.HIDDEN_UNIT
    def forward(self):
        ctx_E = self.ctx_E # 30,50这里的ctx已经经过parameter了
        ctx_V = self.ctx_V
        ctx_S_E = self.ctx_S_E #share
        ctx_S_V = self.ctx_S_V  # share

        # ratio =0.9

        # ctx_x = ctx_E + ctx_V  #这两个域的加一起？
        alpha = 0
        beta = 0 #0.001 0.204
        ctx_E_1 = ctx_E + alpha * ctx_V
        ctx_V_1 = ctx_V + beta * ctx_E

        if ctx_S_E.dim() == 2:#???怎样保证它们共享同一个ebmedding
            ctx_E = ctx_E_1.unsqueeze(0).expand(self.n_cls_E, -1, -1)  # 按不同的在ctx的第0维，进行扩充;将内部，第2维复制为n_cls=37倍，0维和1维不变。  12132,30,50
            ctx_V = ctx_V_1.unsqueeze(0).expand(self.n_cls_V, -1, -1)  # 18388,30,50  等于V本身数量
            ctx_S_E = ctx_S_E.unsqueeze(0).expand(self.n_cls_E, -1, -1)  # 12132,30,50
            ctx_S_V = ctx_S_V.unsqueeze(0).expand(self.n_cls_V, -1, -1)  # 18388,30,50

        prefix_E = self.token_prefix_E  #这里存的是固定的embedding, 先存成item  embedding  12132,1,50
        prefix_V = self.token_prefix_V  # 18388,1,50

        ctx_prefix_E = self.getPrompts(prefix_E,ctx_E,ctx_S_E)#(prefix_E, ctx_E, ctx_S_E) #将共享的拼在一起12132,(1+30+30),50 item,独特，共享
        ctx_prefix_V = self.getPrompts(prefix_V,ctx_V,ctx_S_V)#(prefix_V, ctx_V, ctx_S_V)

        #先将他们进行线性变换，用可学习的Transformer?,这个位置也可以提前，提交到concat之前
        x_E = self.text_encoder_E(ctx_prefix_E) # 在这里进入text_encoder  12132,(1+30+30),50
        x_V = self.text_encoder_V(ctx_prefix_V)

        ctx_prefix_E = x_E
        ctx_prefix_V = x_V

        #下面是求和
        item_embedding = prefix_E.expand(-1, ctx_prefix_E.shape[1],-1)
        prompt_item = torch.cat((ctx_prefix_E, item_embedding), dim=2)
        #把和item concat的结果送给它
        at_wt = self.attention_E(prompt_item)  #这里是计算weight, 先将prompt和item concat，然后经过linear计算它们的相似度
        prompts_E = torch.bmm(at_wt.permute(0, 2, 1), ctx_prefix_E).squeeze()  # 对user加权和之后得到的group emb
        # alpha_E = 0  # 0.9不行  0.01, 0.02都试过
        # prompts_E = prompts_E + alpha_E * prefix_E.squeeze()
        #如果直接求最大呢
        # prompts_E = torch.max(ctx_prefix_E)

        # prompts_E = ratio * prompts_E + (1-ratio) * prefix_E.squeeze()
        # prompts_E = prompts_E + prefix_E.squeeze()

        #aggreage promots V
        item_embedding = prefix_V.expand(-1, ctx_prefix_V.shape[1], -1)
        prompt_item = torch.cat((ctx_prefix_V, item_embedding), dim=2)
        # 把和item concat的结果送给它
        at_wt = self.attention_V(prompt_item)  # 这里是计算weight, 先将prompt和item concat，然后经过linear计算它们的相似度
        prompts_V = torch.bmm(at_wt.permute(0, 2, 1), ctx_prefix_V).squeeze()  # 对user加权和之后得到的group emb


        return prompts_E, prompts_V

    def getPrompts(self, prefix, ctx,ctx_S, suffix=None):#(self, prefix, ctx,ctx_S, suffix=None): ##
        if self.class_token_position == "end":  # 将词前缀+ctx+词后缀拼接，后缀中包含了类名
            prompts = torch.cat(
                [
                    ctx_S, #去掉共享
                    ctx,     # (n_cls, n_ctx, dim) #所有item共享，所有域共享
                    prefix # (n_cls, 1, dim)  #加上item_embedding
                ],
                dim=1,
            )
        #下面不执行
        # elif self.class_token_position == "middle":
        #     half_n_ctx = self.n_ctx // 2
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         # ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        #         # ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,     # (1, 1, dim)
        #                 ctx_i_half1,  # (1, n_ctx//2, dim)
        #                 class_i,      # (1, name_len, dim)
        #                 ctx_i_half2,  # (1, n_ctx//2, dim)
        #                 suffix_i,     # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # elif self.class_token_position == "front":
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i = ctx[i : i + 1, :, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 class_i,   # (1, name_len, dim)
        #                 ctx_i,     # (1, n_ctx, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s) 返回给定输入字符串的标记化表示

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize 要标记的输入字符串或输入字符串列表

        context_length : int
            The context length to use; all CLIP models use 77 as the context length 要使用的上下文长度；所有剪辑模型都使用77作为上下文长度

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length 如果文本编码长于上下文长度，是否截断文本

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        包含结果标记的二维张量，shape=[输入字符串的数量，context_length]
        """
        if isinstance(texts, str):
            texts = [texts]  # text是要标记的输入字符串或输入字符串列表

        sot_token = _tokenizer.encoder["<|startoftext|>"]  # encoder是simpler_tokenizer里边的,sot代表start of text
        eot_token = _tokenizer.encoder["<|endoftext|>"]  # eot代表end of text
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in
                      texts]  # 所有token的格式=开始标志+编码后的text+遍历texts中的[eto_token]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)  # result是一个维度为[所有token长度，上下文长度=77]的0矩阵
        # 为什么_tokenizer.encode(text)是长度=20的？类名不是长度只有1吗？这样的话类名长度是20啊。
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:  # 判断如果token超过了77，要截断
                if truncate:
                    tokens = tokens[:context_length]  # token是0-76+eot_token
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
            # 把result的原始0矩阵填充，每一行都是token的tensor形式,变成[49406, 343, 343343, , 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 16342, 690, 36638, 269, 49407,0,0,0...]一共77个，剩下的都用0填充

        return result  # 结果是start+初始化词+end
class CustomCLIP(nn.Module):
    def __init__(self, cfg, sasr_model_E, num_E, num_V):
        super().__init__()
        self.dev = sasr_model_E.dev
        self.hidden_units = cfg.TRANSFORMER.HIDDEN_UNIT

        # self.concatLayer = PointWiseFeedForward(self.hidden_units*2, self.dropout_rate)
        # self.concatLayer_E = nn.Sequential(
        #     nn.Linear(2* self.hidden_units, self.hidden_units),  # nn.Linear(in_features,out_features)
        #     nn.ReLU(inplace=True),  # new
        #     nn.Dropout(0.2),
        #     nn.Linear(self.hidden_units, sasr_model_E.item_num)
        # )
        # self.concatLayer_V = nn.Sequential(
        #     nn.Linear(2* self.hidden_units, self.hidden_units),  # nn.Linear(in_features,out_features)
        #     nn.ReLU(inplace=True),  # new
        #     nn.Dropout(0.2),
        #     nn.Linear(self.hidden_units, sasr_model_V.item_num)
        # )
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight)
        #     if isinstance(m, nn.Embedding):
        #         nn.init.xavier_normal_(m.weight)

        self.prompt_learner = PromptLearner(cfg, sasr_model_E, num_E, num_V)  #
        # self.tokenized_prompts_E = self.prompt_learner.tokenized_prompts_E  #是为了取不同的item,适应不同item的数量
        # self.tokenized_prompts_V = self.prompt_learner.tokenized_prompts_V
        self.seq_encoder_E = SeqEncoder(sasr_model_E)  # 20220914  init
        # self.seq_encoder_V = SeqEncoder(sasr_model_V)
        # self.image_encoder = clip_model.visual  # visionTransformer 20220914wang
        # self.text_encoder_E = TextEncoder(sasr_model_E)  # 这就是要得到prompt
        # self.text_encoder_V = TextEncoder(sasr_model_V)
        # self.logit_scale = sasr_model.logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.log2feats = sasr_model.log2feats
        # self.dtype = sasr_model.dtype
        # self.embedding_seq = self.prompt_learner.seq_emb
        # embedding_seq = self.embedding_seq
        self.item_embeddings_E = sasr_model_E.item_emb  # 这里需要取区间，否则是全部的 ##############################
        # self.item_embeddings_V = sasr_model_V.item_emb

    def forward(self, seq_E, seq_V, label_E, label_V, mode):  #seq要变成网前补0
        prompts_E, prompts_V = self.prompt_learner()  #forward 拼接后的embedding prompt # class*77*512
        # logit_scale = self.logit_scale.exp()  # exp()返回x的指数,tensor(100., device='cuda:1')
        loss_ssl_E = 0
        loss_ssl_V = 0
        text_features_E = prompts_E  # 12132,50
        text_features_V = prompts_V
        logits_E =0
        logits_V =0
        if not seq_E is None:
        # if mode == "train":
            seq_features_E = self.seq_encoder_E(seq_E)  # 128,64,50
            final_feat_E = seq_features_E[:, -1, :] #(128,50)
            logits_E = text_features_E.matmul(final_feat_E.unsqueeze(-1)).squeeze(-1) #(12669,50)-128,50,1-128,12669,1-128,12669
            # seq_features_V = self.seq_encoder_V(seq_V)
            # final_feat_V = seq_features_V[:, -1, :]
            # concat_output = torch.cat([final_feat_E, final_feat_V], dim=1)
            # pred_E = self.concatLayer_E(final_feat_E)
            # logits_E += pred_E
            if mode =="train":  #主要是为了SSL
                # seq_features_V = self.seq_encoder_V(seq_V)
                # final_feat_V = seq_features_V[:, -1, :]
                # final_feat_E += final_feat_V
                # concat_output = torch.cat([final_feat_E, final_feat_V], dim=1)
                # logits_E = text_features_E.matmul(final_feat_E.unsqueeze(-1)).squeeze(-1)
                # pred_E = self.concatLayer_E(final_feat_V)
                # logits_E += pred_E
                prompt_label_E = text_features_E[label_E]  # 取出label对应的prompt
                # loss_ssl_E = self.SSL(prompt_label_E, final_feat_E) #/ logit_scale
        if not seq_V is None:
        # if mode == "train":
            seq_features_V = self.seq_encoder_E(seq_V)
            final_feat_V = seq_features_V[:, -1, :]
            logits_V = text_features_V.matmul(final_feat_V.unsqueeze(-1)).squeeze(-1)
            # seq_features_E = self.seq_encoder_E(seq_E)  # 128,50,50
            # final_feat_E = seq_features_E[:, -1, :]
            # concat_output = torch.cat([final_feat_E, final_feat_V], dim=1)
            # pred_V = self.concatLayer_V(final_feat_V)
            # logits_V += pred_V
            if mode == "train":#为了SSL
                # seq_features_E = self.seq_encoder_E(seq_E)  # 128,50,50
                # final_feat_E = seq_features_E[:, -1, :]
                # final_feat_V += final_feat_E
                # concat_output = torch.cat([final_feat_E, final_feat_V], dim=1)
                # logits_V = text_features_V.matmul(final_feat_V.unsqueeze(-1)).squeeze(-1)
                # pred_V = self.concatLayer_V(concat_output)
                # logits_V += pred_V
                prompt_label_V = text_features_V[label_V]  # 取出label对应的prompt
                # loss_ssl_V = self.SSL(prompt_label_V, final_feat_V)# / logit_scale

        return logits_E, logits_V, loss_ssl_E, loss_ssl_V # 得到logits，相似度概率

    #对比学习代码
    def SSL(self, prompt_features, seq_features):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos_E = score(prompt_features, seq_features)
        neg1_E = score(prompt_features, row_column_shuffle(prompt_features))

        # neg2 = score(sess_emb_lgcn)
        one = torch.FloatTensor(neg1_E.shape[0]).fill_(1).to(neg1_E.device)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos_E))-torch.log(1e-8 + (one - torch.sigmoid(neg1_E))))
        return con_loss
    #计算两个prompt之间的对比
    def SSL2(self, prompt_E, seq_feature_E, prompt_V):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos_E = score(prompt_E, seq_feature_E)
        # pos_V = score (prompt_V, seq_feature_V)
        # pos =pos_E + pos_V
        pos = pos_E
        # neg1_E = score(prompt_E, row_column_shuffle(prompt_E))
        neg2 = score(prompt_E, prompt_V)
        neg = neg2

        # neg2 = score(sess_emb_lgcn)
        one = torch.FloatTensor(neg.shape[0]).fill_(1).to(neg.device)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg))))
        return con_loss

@TRAINER_REGISTRY.register()  # 要注册的对象
class CoOp(TrainerX):
    """Context Optimization (CoOp). 上下文优化（CoOp）。

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        # classnames = self.dm.dataset.classnames  # 数据集的类名

        # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")  # 这里是Loading trainer：CoOp后的输出内容：RN50
        sasr_model_E = load_clip_to_cpu(cfg)#cfg要两个  # 加载clip模型，在load_clip_to_cpu函数中
        # 将SASR模型初始化完毕
        # if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
        #     CLIP's default precision is fp16 CLIP的默认精度为fp16
        #     sasr_model.float()

        print("Building custom SASRec_E, SASRec_V")  #这里的model是指prompt learning中的参数
        self.model = CustomCLIP(cfg, sasr_model_E,self.num_classes_E, self.num_classes_V) #init
        # self.model_V = CustomCLIP(cfg, sasr_model_V)
        # 进入Custom CLIP，将SASR中的东西在CustomCLIP中进行初始化

        # 完成了初始化context，并且将其进入编码器，生成编码，之后再进行vector转换，变成tensor的embedding##############################
        # 通用的text都变成了[sot X X X ...classs token eot]编码变成对应数字[49406,343,...,5777,575,49407]之后变成77长度的，不够的用0填充。整个再变成初始权重的embedding
        print("Turning off gradients in both the image and the text encoder")  # “关闭图像和文本编码器中的梯度
        for name, param in self.model.named_parameters():  #
            if "prompt_learner" not in name:  #如果参数不是prompt learner,就将梯度更新设为false
                param.requires_grad_(False)  # 也就是训练时不更新
            if "prompt_learner" not in name and "concatLayer" in name:
                param.requires_grad_(True)
        # for name, param in self.model_V.named_parameters():  # 返回参数名字+参数。返回模块参数上的迭代器，生成参数的名称以及参数本身。
        #     if "prompt_learner" not in name:  # 除了prompt_learner参数梯度保持True，模型中的其他参数，需要梯度的参数就是false
        #         param.requires_grad_(False)  # 也就是训练时不更新
        # if cfg.MODEL.INIT_WEIGHTS:  # 如果模型的初始化参数存在，就执行下面，实际上是不存在。INIT_WEIGHTS：‘’
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)  # 模型配备完毕，到设备上执行？
        # self.model_V.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)  # 用于构建优化器的函数包装器。将得到的prompt文本按参数选择的优化器进行优化，返回的是优化器=torch.optim.SGD

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)  #设置学习率的更新方，余弦退火 从这里进入test的注册。预热学习率的函数，用torch.optim.lr_scheduler.CosineAnneadlingLR,余弦退火顺序调整，进行lr的优化，scheduler=ConstantWarmupScheduler
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        # 这句不太知道什么意思，注册模型，将self中存在model\optim\sched，放进参数中？这种形式：_models[name] = model
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None  # 梯度缩放，这句不用，因为PREC=‘fp16’
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)  # 在两个GPU上并行计算，数据平均分配，两个GPU分别向前传播，最后结果汇总到一个GPU上
    def forward_backward_domain(self, batch_x, type):
        if type=='source':
            seq_E, label_E = self.parse_batch_train(batch_x)
            seq_V, label_V = None,None
        else:
            seq_V, label_V = self.parse_batch_train(batch_x)
            seq_E, label_E = None,None
        # 这里的image=input=batch[seq]
        prec = self.cfg.TRAINER.COOP.PREC  # fp16,是这个格式的
        if prec == "amp": #不执行
            with autocast():
                output_E = self.model(seq_E)
                output_V = self.model(seq_V)
                loss_E = F.cross_entropy(output_E, label_E)
                loss_V = F.cross_entropy(output_V, label_V)

                loss = loss_E + loss_V
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # output = self.model(image)  # （32，37）得到表示，输出预测类别的索引 20220914
            output_E, output_V, loss_ssl_E, loss_ssl_V = self.model(seq_E, seq_V, label_E, label_V,mode='train')  # 这里指的prompt learner-->forward（32，37）得到表示，输出预测类别的索引 20220914  output（256，3368）（batch_size,n_class）20220921
            # output_V = self.model(seq_V,"V")
            if type == 'source':
                loss = F.cross_entropy(output_E, label_E)  # 交叉熵损失，将得到的输出张量与label之间做loss，32是batch大小，每次处理32张图片，每张图都与37个类去计算
                loss_ssl = loss_ssl_E
            else:
                loss = F.cross_entropy(output_V, label_V)
                loss_ssl = loss_ssl_V
            beta = 0.001 #0.001 #0.001 #0.5  增加对比学习会变好
            # beta_ssl_V = 0.01#1.05  # 0.05
            # beta_V = 1.05 #0.1
            # beta = 0 # 就是去掉对比学习
            # loss = (loss_E + loss_V) + beta * (loss_ssl_E + loss_ssl_V)
            losses = loss + beta * loss_ssl
            # loss = (loss_E + beta_V*loss_V) + (beta * loss_ssl_E + beta_ssl_V*loss_ssl_V)
            self.model_backward_and_update(losses)  # 损失更新

        loss_summary = {
            "loss": losses.item(),
            # "acc": compute_accuracy(output, label)[0].item(),  # 计算loss和准确率
            # "NDCG": evaluate_test(output, label)  # 用户u对i的兴趣分数，用户u对i的真实分数
        }

        if (self.batch_idx + 1) == self.num_batches:  # 如果这一epoch的batch都走了一遍了，就更改学习率，用学习率优化的参数scheds[name].step()
            self.update_lr() #当前epoch中所有的batch跑完，再更新loss

        return loss_summary
    def forward_backward(self, batch_E, batch_V, type):
        seq_E, label_E = self.parse_batch_train(batch_E)
        seq_V, label_V = self.parse_batch_train(batch_V)
        # 这里的image=input=batch[seq]
        prec = self.cfg.TRAINER.COOP.PREC  # fp16,是这个格式的
        if prec == "amp": #不执行
            with autocast():
                output_E = self.model(seq_E)
                output_V = self.model(seq_V)
                loss_E = F.cross_entropy(output_E, label_E)
                loss_V = F.cross_entropy(output_V, label_V)

                loss = loss_E + loss_V
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # output = self.model(image)  # （32，37）得到表示，输出预测类别的索引 20220914
            output_E, output_V, loss_ssl_E, loss_ssl_V = self.model(seq_E, seq_V, label_E, label_V,"train")  # 这里指的prompt learner-->forward（32，37）得到表示，输出预测类别的索引 20220914  output（256，3368）（batch_size,n_class）20220921

            loss_E = F.cross_entropy(output_E, label_E)  # 交叉熵损失，将得到的输出张量与label之间做loss，32是batch大小，每次处理32张图片，每张图都与37个类去计算
            loss_V = F.cross_entropy(output_V, label_V)
            # loss_V =0
            # loss_ssl_V= 0
            # loss_E =0
            # loss_ssl_E =0

            beta = 0.001 #0.001 #0.001 #0.5  增加对比学习会变好
            # beta_ssl_V = 0.01#1.05  # 0.05
            # beta_V = 1.05 #0.1
            # beta = 0 # 就是去掉对比学习
            # loss = (loss_E + loss_V) + beta * (loss_ssl_E + loss_ssl_V)
            # loss = (loss_E + beta_V*loss_V) + (beta * loss_ssl_E + beta_ssl_V*loss_ssl_V)
            loss_E = loss_E + loss_V + beta * (loss_ssl_E + loss_ssl_V)
            # loss_V += beta * loss_ssl_V
            self.model_backward_and_update(loss_E)  # 损失更新
            # self.model_backward_and_update(loss_V)  # 损失更新

        loss_summary = {
            "loss_E": loss_E.item(),
            # "loss_V": loss_V.item()
            # "acc": compute_accuracy(output, label)[0].item(),  # 计算loss和准确率
            # "NDCG": evaluate_test(output, label)  # 用户u对i的兴趣分数，用户u对i的真实分数
        }

        if (self.batch_idx + 1) == self.num_batches:  # 如果这一epoch的batch都走了一遍了，就更改学习率，用学习率优化的参数scheds[name].step()
            self.update_lr() #当前epoch中所有的batch跑完，再更新loss

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["seq"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
