import numpy as np
import torch
from torch import nn

import arguments


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
def convert_weights(model: nn.Module):  # 将适用的模型参数转换为fp16
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)  # 递归的将这个函数应用于每个子模块

class SASRec(torch.nn.Module):
    def __init__(self, item_num, dev, hidden_units, maxlen, dropout_rate, num_heads, num_blocks):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = dev
        self.hidden_units = hidden_units

        self.maxlen = maxlen
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num, self.hidden_units, padding_idx=0)  # (3417,50)#############这里有问题
        self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                            self.num_heads,
                                                            self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
    #这里是得到句子的表示
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # 先经过embedding(3417,50)
        seqs *= self.item_emb.embedding_dim ** 0.5  # 50**0.5,50开根号，三维seq张量*一个数，数值变化，维度不变
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # np.tile(array(range（200）),[128,1])把array(range(200))横向纵向的赋复制为(128,1)
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # 时间线掩码，将0的列全部变成Ture，不等于0的变为False（128，200）
        seqs *= ~timeline_mask.unsqueeze(-1)  # 在最后一个维度扩充broadcast in last dim(128,200)--(128,200,1),seqs：（128，200，5）--（128，200，5）
        # ~布尔值反转，也就是0=False，不是0的是True。
        tl = seqs.shape[1]  # tl=200 time dim len for enforce causality强化因果关系的时间维度
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # 生成200,200维度的布尔类型
        # （200,200）布尔值下三角矩阵，再翻转值True-False
        for i in range(len(self.attention_layers)):  # range(0,2),这里是多头注意力，只有两层。可以像常规Python列表一样索引，但是它包含的模块已正确注册，所有人都可以看到
            seqs = torch.transpose(seqs, 0, 1)  # 在seqs的（0，1）维进行转置
            Q = self.attention_layernorms[i](seqs)  # 注意力前layernorm，Q是最终的x
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)  # 注意力网络
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs  # Q+output(Q)：浅绿色x+浅粉色z
            seqs = torch.transpose(seqs, 0, 1)  # （200，128，50）

            seqs = self.forward_layernorms[i](seqs)  # 将残差后的数值送入Laynorm，得到整个句子经过多头注意力的特征
            seqs = self.forward_layers[i](seqs)  # 前馈神经网络
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C) 输入一个张量，得到
        # 这里放在每个forward之后。每个句子有一个输入对应一个输出
        return log_feats
    #得到正例和负例的表示
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training u,seq,pos,neg
        log_feats = self.log2feats(log_seqs)   # (128,200,50) # seq is used   user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))   # (128,200,50) # pos is used
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))  # neg is used

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (128,200)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # 原本是(128，200，50)在最内维求和之后变为（128，200）

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

def build_model(state_dict: dict, cfg):
    # vit = "visual.proj" in state_dict  #
    #
    # if vit:
    #     vision_width = state_dict["visual.conv1.weight"].shape[0]
    #     vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    #     vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    #     grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    #     image_resolution = vision_patch_size * grid_size
    # else:  # 如果visual.proj不在里边，
    #     counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]  # 计算层数？是以layer1，2，3，4开头的属性下有多少个子层，比如1.0；1.1；counts:[3,4,6,3] ，visual.layer1,2,3,4层，list是以visual.layer1,2,3,4开头的参数，用.分开后，看第三个字符，即：visual/layer1/0，
    #     vision_layers = tuple(counts)  # vision_layers=(3,4,6,3) 视觉层有这些
    #     vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # 64 视觉层宽度
    #     output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)  # 7 输出宽度
    #     vision_patch_size = None  # 视觉patch大小，patch可以理解为图像块
    #     assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]  # 判断前边是否等于后边？True：7^2+1==50
    #     image_resolution = output_width * 32  # 图像分辨率
    hidden_units = state_dict["item_emb.weight"].shape[1]
    maxlen = state_dict["pos_emb.weight"].shape[0]
    item_num = state_dict["item_emb.weight"].shape[0]
    dev = cfg.TRANSFORMER.DEVICE
    dropout_rate = cfg.TRANSFORMER.DROPOUT_RATE  #0.02  # 可以写死吗
    num_heads = cfg.TRANSFORMER.NUM_HEADS
    num_blocks = cfg.TRANSFORMER.NUM_BLOCKS


    # embed_dim = state_dict["text_projection"].shape[1]
    # context_length = state_dict["positional_embedding"].shape[0]  # 200
    # vocab_size = state_dict["token_embedding.weight"].shape[0]
    # transformer_width = state_dict["ln_final.weight"].shape[0]
    # transformer_heads = transformer_width // 64
    # transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    # 以上都是从VIT文件中load参数进去，是网络上训练好的参数，不用再训练了。
    model = SASRec(item_num, dev, hidden_units, maxlen, dropout_rate, num_heads, num_blocks)  #dropout_rate, num_heads, num_blocks)
    #     embed_dim,
    #     image_resolution, vision_layers, vision_width, vision_patch_size,
    #     context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    # )

    # for key in ["input_resolution", "context_length", "vocab_size"]:  # 输入清晰度、上下文长度、字典size
    #     if key in state_dict:
    #         del state_dict[key]  # 用于删除元素！真的给删了

    # convert_weights(model)  # 将模型的每一层都转换成fp16的格式   guo
    model.load_state_dict(state_dict)  # 将参数和缓冲区从：attr:`state_dict`复制到此模块及其子模块。如果：attr:'strict'为'True'，则attr:`state_dict`的键必须与返回的键完全匹配通过该模块的：meth:`~torch.nn.module。state_dict’函数。
    return model.eval()  # 将模块设置为评估模式。默认是评估模式的，如果要训练，参数再改为train
