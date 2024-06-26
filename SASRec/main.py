import os
import time
import torch
import argparse

from model import SASRec
from utils import *
from arguments import *
import datasets.oxford_pets
from dassl.data import DataManager
from dassl.config.defaults import *
from dassl.config import get_cfg_default


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def extend_cfg(cfg):
    """
    Add new config variables. 命令行参数解析方法

    E.g.
        from yacs.config import CfgNode as CN  来自yacs。配置将CfgNode导入为CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN  # yacs库用于为系统构建config文件

    cfg.TRAINER.COOP = CN()  # 在这调用coop.py??  当COOP是trainer时
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors 上下文向量的数量
    cfg.TRAINER.COOP.CSC = False  # class-specific context 类特定上下文
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words 初始化词
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front' “中间”或“末端”或“前部”
    # 以上是将COOP训练器的参数配置好

    cfg.TRAINER.COCOOP = CN()  # 当COCOOP是trainer时，配置他的参数
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def reset_cfg(cfg, args):  # 将args的参数赋值给cfg
    if args.root:  # 输入参数，就是我们在debug中配置的那些，root=DATA
        cfg.DATASET.ROOT = args.root

    if args.output_dir:  # 输出的文件路径应该在哪
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:  # ''
        cfg.RESUME = args.resume

    if args.seed:  # 1
        cfg.SEED = args.seed

    if args.source_domains:  # None
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:  # None
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:  # None
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:  # CoOp
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:  # ''
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:  # ''
        cfg.MODEL.HEAD.NAME = args.head


def setup_cfg(args):
    cfg = get_cfg_default()  # 用于yaml的配置文件
    extend_cfg(cfg)
    # 到此为止，都还是配置，没有选定用哪个trainer
    # 1. From the dataset config file  数据配置文件 configs/datasets/oxford_pets.yaml
    root = os.path.abspath(os.path.dirname(os.getcwd()))
    if args.dataset_config_file:  # 如果这个数据集config文件存在，则进行以下操作
        cfg.merge_from_file(os.path.join(root, args.dataset_config_file))
        # 处理不同实验中的不同超参设置时，用这个函数，会比较每个实验特有的config与默认参数的区别，会将默认参数与特定参数不同的部分，用特定参数覆盖

    # 2. From the method config file  方法配置文件 configs/trainers/CoOp/rn50_ep50.yaml
    if args.config_file:
        cfg.merge_from_file(os.path.join(root, args.config_file))  # 将默认参数与特定参数不同的部分，用特定参数覆盖；用文件中的参数配置去覆盖之前配置的参数
    # 经过这一步之后，就将模型的名字传进去了，BACKBONE NAME = RN50

    # 3. From input arguments  输入参数
    reset_cfg(cfg, args)  # 将args的参数赋值给cfg

    # 4. From optional input arguments 来自可选输入参数
    cfg.merge_from_list(args.opts)  # 用列表的方式进行不同参数的修改

    cfg.freeze()  # 将修改好的参数冻结，不再变动

    return cfg


#
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True)
# parser.add_argument('--train_dir', required=True)
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=20, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.5, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)
# parser.add_argument('--device', default='cpu', type=str)
# parser.add_argument('--inference_only', default=False, type=str2bool)
# parser.add_argument('--state_dict_path', default=None, type=str)

args = args_S()


# if not os.path.isdir(args.dataset + '_' + args.train_dir):
#     os.makedirs(args.dataset + '_' + args.train_dir)
# with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()
# 将字符型转为int,并将label加入seq,以便用于sampler
def getData(dataset):
    seqList_t = []
    for items in dataset:
        seqList = []
        seq = items.session.split(",")
        label = items.label
        seq.append(label)
        for item in seq:
            seqList.append(int(item))

        seqList_t.append(seqList)

    return seqList_t


# 返回用于测试的数据
def getTestData(dataset, data_set):
    seqList_t = []
    labels = []
    for items in dataset:
        seqList = []
        seq = items.session.split(",")
        label = items.label
        labels.append([label])
        # seq.append(label)
        for item in seq:
            seqList.append(int(item))

        seqList_t.append(seqList)

    return seqList_t, labels, len(seqList_t), data_set.num_classes_E


# 训练好的配置文件在sasrec.py中，让coop读出
# 数据集在datasets/oxfrod_pets.py中
# 参数配置在configs/rn50...yaml中
# 训练好的模型保存在SASRec中
if __name__ == '__main__':
    # global dataset
    cfg = setup_cfg(args)
    dm = DataManager(cfg)
    data_set = dm.dataset
    train = getData(data_set.train_x_E)
    test = getTestData(data_set.test_E, data_set)
    valid = getTestData(data_set.val_E, data_set)

    # train_loader_x = dm.train_loader_x
    # train_loader_u = dm.train_loader_u  # optional, can be None
    # val_loader = dm.val_loader  # optional, can be None
    # test_loader = dm.test_loader

    # num_classes = dm.num_classes
    # num_source_domains = dm.num_source_domains
    # lab2cname = dm.lab2cname  # dict {label: classname}

    # dataset = data_partition(args.dataset)

    # [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)  num_batch=47
    # cc = 0.0
    # for u in user_train:
    #     cc += len(user_train[u])  # train中每条用户item的长度，总共6040个用户，3416个item，cc=987531.0个item遍历
    # print('average sequence length: %.2f' % (cc / len(user_train)))  # 平均序列长度

    f = open(os.path.join(os.getcwd(), 'log.txt'), 'w')  # 打开name='ml-1m_default/log.txt' ，模式写入

    sampler = WarpSampler(train, len(train), data_set.num_classes_E, batch_size=args.batch_size, maxlen=args.maxlen,
                          n_workers=3)

    # args = arguments.args_S()
    # self.user_num = user_num
    item_num = data_set.num_classes_E + 30  # cfg.DATALOADER.NUM_CLASS  E-domain
    # item_num = data_set.num_classes + 1 #V-domain
    dev = cfg.TRANSFORMER.DEVICE
    hidden_units = cfg.TRANSFORMER.HIDDEN_UNIT
    maxlen = cfg.TRANSFORMER.MAX_LEN
    dropout_rate = cfg.TRANSFORMER.DROPOUT_RATE
    num_heads = cfg.TRANSFORMER.NUM_HEADS
    num_blocks = cfg.TRANSFORMER.NUM_BLOCKS

    model = SASRec(item_num, dev, hidden_units, maxlen, dropout_rate, num_heads, num_blocks).to(dev)  # 8
    # model = SASRec(cfg, cfg.DATALOADER.NUM_CLASS).to(args.device) # no ReLU activation in original SASRec implementation?原始SASRec实现中无ReLU激活?
    # 给出网络层的名字和参数的迭代器
    for name, param in model.named_parameters():  # 读模型网络层名字+参数
        try:  # para(3417,50) name:'item_emb.weight'
            torch.nn.init.xavier_normal_(param.data)  # 初始化权重
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training，设为train mode

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, test, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRANSFORMER.LR, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition

        # for batch_idx, batch in enumerate(train_loader_x):
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=dev), torch.zeros(neg_logits.shape, device=dev)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                             loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating test', end='')  # 将val  test都一块用了，val也就失去意义了
            t_test = evaluate(model, test, args)
            print('\nEvaluating validate', end='')
            t_valid = evaluate(model, valid, args)
            print('\n epoch:%d, time: %f(s), valid (NDCG@20: %.4f, NDCG@50: %.4f, HR@20: %.4f, HR@50: %.4f),'
                  ' test (NDCG@20: %.4f, NDCG@50: %.4f, HR@20: %.4f, HR@50: %.4f)'
                  % (
                  epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2], t_test[3]))
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            # folder = args.dataset + '_' + args.train_dir
            folder = os.path.os.getcwd()
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}_E.pth'
            # fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}_V.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
    # 训练好的配置文件在sasrec.py中，让coop读出
    # 数据集在datasets/oxfrod_pets.py中
    # 参数配置在configs/rn50...yaml中
    # 训练好的模型保存在SASRec中
