# import argparse
#

import argparse
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def args_S():
    parser = argparse.ArgumentParser()  # 创建解析器，ArgumentParser包含将命令行解析成python数据类型所需的全部信息
    parser.add_argument("--root", type=str, default="", help="path to dataset")  # 调用add_argument方法添加参数  数据集的路径
    parser.add_argument("--output-dir", type=str, default="", help="output directory")  # 输出目录
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",  # 检查点目录（从中恢复培训）
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"  # 只有正值才能启用固定种子
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"  # DA/DG的源域
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"  # DA/DG的目标域
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"  # 数据扩充方法
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"  # 配置文件的路径
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",  # 数据集设置的配置文件路径
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")  # trainer的name
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")  # CNN主干网名称
    parser.add_argument("--head", type=str, default="", help="name of head")  # head的name
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")  # 仅评估
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",  # 从该目录加载模型，用于仅评估模式，也就是先下载，存到这里边？训练时再从里边得到？
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"  # 用于评估的该时期的负荷模型权重
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"  # 是否不调用trainer.train（）
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",  # 使用命令行修改配置选项
    )
    # parser.add_argument("--batch_size", type=int, default=128, help="path to dataset")
    parser.add_argument("--device", type=str, default='cuda', help="output directory")
    # parser.add_argument("--dropout_rate", type=float, default=0.2)
    # parser.add_argument("--hidden_units", type=int, default=512)
    # parser.add_argument("--inference_only", type=bool, default=False)
    # parser.add_argument("--l2_emb", type=float, default=0.0)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--maxlen", type=int, default=77)
    # parser.add_argument("--num_blocks", type=int, default=2)
    # parser.add_argument("--num_epoch", type=int, default=20)
    # parser.add_argument("--num_heads", type=int, default=1)
    # parser.add_argument("--embed_dim",type=int,default=1024)
    # parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=64, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0, type=float) #0.5
    parser.add_argument('--l2_emb', default=0.0, type=float)
    # parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)#type=str2bool
    parser.add_argument('--state_dict_path', default=None, type=str)
    # parser.add_argument('--dataset', default='ml-1m')
    args = parser.parse_args()  # 使用parse_args()解析添加的参数

    return args