import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
#上面这些import可能引起注册的问题
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from arguments import *

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())  # 将args字典形式中的key形成一个列表，这里存储的即为“参数名称”
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

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

    if args.backbone:  #''
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:  # ''
        cfg.MODEL.HEAD.NAME = args.head


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


def setup_cfg(args):
    cfg = get_cfg_default()  # 用于yaml的配置文件
    extend_cfg(cfg)
    # 到此为止，都还是配置，没有选定用哪个trainer
    # 1. From the dataset config file  数据配置文件 configs/datasets/oxford_pets.yaml
    if args.dataset_config_file:  # 如果这个数据集config文件存在，则进行以下操作
        cfg.merge_from_file(args.dataset_config_file)
        # 处理不同实验中的不同超参设置时，用这个函数，会比较每个实验特有的config与默认参数的区别，会将默认参数与特定参数不同的部分，用特定参数覆盖

    # 2. From the method config file  方法配置文件 configs/trainers/CoOp/rn50_ep50.yaml
    if args.config_file:
        cfg.merge_from_file(args.config_file)  # 将默认参数与特定参数不同的部分，用特定参数覆盖；用文件中的参数配置去覆盖之前配置的参数
    # 经过这一步之后，就将模型的名字传进去了，BACKBONE NAME = RN50

    # 3. From input arguments  输入参数
    reset_cfg(cfg, args)  # 将args的参数赋值给cfg

    # 4. From optional input arguments 来自可选输入参数
    cfg.merge_from_list(args.opts)  # 用列表的方式进行不同参数的修改
    cfg.freeze()  # 将修改好的参数冻结，不再变动
    return cfg

def main(args):
    cfg = setup_cfg(args)  # 将args中带着的参数，经过处理后给cfg
    if cfg.SEED >= 0:  # 如果cfg的种子>=0,是的，他=1,2,3...
        print("Setting fixed seed: {}".format(cfg.SEED))  # 设置固定种子值
        set_random_seed(cfg.SEED)  # 设置随机种子，也就是设置种子，用这个函数，让调用种子的地方可以利用seed=1的值
    setup_logger(cfg.OUTPUT_DIR)  # 信息记录系统作用：将终端显示的内容保存到文件中，保存到output/../..log.txt中

    if torch.cuda.is_available() and cfg.USE_CUDA:  # 使用CUDA cuda是否可用且USE_CUDA=True
        torch.backends.cudnn.benchmark = True  # 这句话会增加运行效率

    # print_args(args, cfg)  # 输出参数配置  这里没输出的有：resume（本身cfg中就没有）、backbone(cfg中有也没有输出，为什么，后面输出吗？)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))  # 得到环境信息
    #dataset 写在了datasets/oxford_flowers.py中，配置文件写在了两个yaml文件中
    trainer = build_trainer(cfg)  # dassl/engine/trainer-> SimpleTrainer-->__init__ 转向coop中的build_model

    # trainer.test() #for test guo
    if args.eval_only:  # false
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test() #trainer-test()
        return

    if not args.no_train:
        trainer.train()  # 将之前得到的trainer进行train
        # 在trainer下的train()里进行通用训练循环，参数是start_epoch和max_epoch

    # if args.no_train is False:
    #     print("Start training...")
    #     trainer.train()
    #     print("Train over.")
    #     trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #     print("Already load the best recommender. Start testing...")
    #     trainer.test()

if __name__ == "__main__":
    args = args_S()
    main(args)  # 将args放入main函数中
#训练好的配置文件在SASRec/sasrec中，让coop读出
#数据集在datasets/oxfrod_pets.py中  同时加载两个域
#参数配置在configs/rn50...yaml中
#训练好的模型保存在SASRec中
#训练之前需要将see3/prompt_learner删除
#测试代码在dassl/engine/trainer.py->test
#上面这些import可能引起注册的问题