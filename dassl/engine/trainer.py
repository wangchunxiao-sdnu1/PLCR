import copy
import sys
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
import arguments
from arguments import args_S as args

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    由CNN主干组成的简单神经网络
以及可选的头部，例如用于分类的mlp
    """
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(  # backbone名字、是否输出日志、是否已经预训练
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):  # 进入分类器部分
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):  # mode:'eval',
        names = self.get_model_names(names)  # names="prompt_learner"

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()  # 进入dassl.module中的eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:  # 两个都是None
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops.通用训练循环。"""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()#如果存在已经训练好的模型，就恢复一下
        #预训练source
        # num =10
        # for i in range(0, 20):
        #     if i==0: #只在初始化时交互

        # for self.epoch in range(self.start_epoch, 200):  # （0，50）start=0,end=50,step=1
        #     self.before_epoch()  # 20220905 no use
        #     self.run_epoch_domain('source')
        #     self.after_epoch()
        #     if self.epoch %10 ==0:
        #         self.after_train("source")
            #预训练target
        best_ndcg_E =0
        best_ndcg_V =0
        for self.epoch in range(self.start_epoch, 200):  # （0，50）start=0,end=50,step=1
            self.before_epoch() #
            self.run_epoch_domain("source")
            self.run_epoch_domain("target")
            # self.run_epoch()
            if self.epoch % 1 == 0:
                ndcg_E, ndcg_V = self.after_train("all")
                if ndcg_E> best_ndcg_E:
                    best_ndcg_E = ndcg_E
                if ndcg_V >= best_ndcg_V:
                    best_ndcg_V = ndcg_V
            print ("the best ndcg in E and V:", best_ndcg_E, best_ndcg_V)
            self.after_epoch()
        # for self.epoch in range(self.start_epoch, 100):  # （0，50）start=0,end=50,step=1
        #     self.before_epoch()  # 20220905 no use
        #     self.run_epoch_domain("source")
        #     self.run_epoch_domain('target')
        #     self.after_epoch()
        #     if self.epoch %10 ==0:
        #         self.after_train("source")
        #         self.after_train("target")

        #预训练完之后，再joint训练
            # for self.epoch in range(self.start_epoch, 10):  # （0，50）start=0,end=50,step=1
            #     self.before_epoch()  # 20220905 no use
            #     if True:
            #         self.run_epoch_domain('source')
            #         if self.epoch > 30 and self.epoch % 10 == 0:
            #             self.after_train("source")
            #     if True:
            #         self.run_epoch_domain('target')
            #         if self.epoch > 30 and self.epoch % 10 == 0:
            #             self.after_train("target")
            #     if self.epoch>1 and self.epoch % 3 ==0:  #join一下
            #         self.run_epoch()
            #     if self.epoch > 30 and self.epoch % 10 == 0:
            #         self.after_train("all")
            #     self.after_epoch()

        # self.after_train("all")  #这里最后的输出


    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def run_epoch_domain(self, type):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch_E, batch_V, type):
        raise NotImplementedError
    def forward_backward_domain(self, batch, type):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions.一个实现通用函数的简单训练器类"""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = cfg.TRANSFORMER.DEVICE#torch.device("cuda")
            # self.device = 'cuda:1'
        # else:
        #     self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader() #这里还没有补0
        self.build_model()
        self.evaluator = build_evaluator(cfg) #转向dassl/evluation/evaluator.py
        self.best_result = -np.inf
        # self.item_emb = torch.nn.Embedding(4000, 512, padding_idx=0).to(self.device)  # (3417,50)
        # self.args = args()

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)  
        # 加载数据集+build transform（判断用自定义transform进行test或train）+构建train_loader_x

        # self.dataset = dm.dataset
        self.train_loader_x_E = dm.train_loader_x_E
        self.train_loader_x_V = dm.train_loader_x_V
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader_E = dm.val_loader_E  # optional, can be None
        self.val_loader_V = dm.val_loader_V
        self.test_loader_E = dm.test_loader_E
        self.test_loader_V = dm.test_loader_V

        self.num_classes_E = dm.num_classes_E
        self.num_classes_V = dm.num_classes_V
        self.num_source_domains = dm.num_source_domains
        self.lab2cname_E = dm.lab2cname_E  # dict {label: classname}
        self.lab2cname_V = dm.lab2cname_V

        self.dm = dm

    def build_model(self):
        """Build and register model.构建和注册模型

        The default builds a classification model along with its默认情况下，将构建一个分类模型及其优化器和调度器。
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary. 如有必要，定制培训师可以重新实施此方法。
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME: #不执行
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer 初始化摘要编写器
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time) 记住开始时间（用于计算经过的时间）
        self.time_start = time.time()


    def after_train(self, domain):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST  # 两个epoch之后就进行这里，true
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":#不执行
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
                #
            else:  # ==last_step
                print("Deploy the last-epoch model")
            if domain=="all":
                ndcg10_E= self.test(self.val_loader_E, self.test_loader_E, self.test_loader_V, "test", "E-domain")#,epoch)  # 需要训练的epoch进行完了之后，再去进行test,全部数据
                ndcg10_V= self.test(self.val_loader_V, self.test_loader_E, self.test_loader_V, "test", "V-domain")#,epoch)
            if domain=="source":
                self.test(self.val_loader_E, self.test_loader_E, "test", "E-domain")  # ,epoch)
            if domain == "target":
                self.test(self.val_loader_V, self.test_loader_V, "test", "V-domain")  # ,epoch)
            # self.test(self.dm.dataset.val_E, self.dm.dataset.test_E, "test", "E-domain")
            # self.test(self.dm.dataset.val_V, self.dm.dataset.test_V, "test", "V-domain")

        # Show elapsed time  显示已用时间
        elapsed = round(time.time() - self.time_start)  # time.time()得到的是1970年到当前的秒数，单位是秒，不是毫秒。当前-进入train之前
        elapsed = str(datetime.timedelta(seconds=elapsed))  # 将全部的秒数，读成几小时几分几秒的形式，比如60s=1分钟
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

        return ndcg10_E, ndcg10_V

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch  # 判断是否等于最大epoch-1，如果是，则last_epoch=1,否则=0  第一轮=False 第二轮=Turue
        do_test = not self.cfg.TEST.NO_TEST  # 如果是test的话 do_best=true
        meet_checkpoint_freq = (  # 满足checkpoint的freq频率？
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )  # False#False

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":  # True and final_Model=last_epoch，第一轮不走
            # curr_result = self.test(split="val")
            curr_result= self.test(self.val_loader_E, self.test_loader_E, "test", "E-domain")  # 所有epoch进行完了之后，再去进行test
            # self.test(self.val_loader_V, self.test_loader_V, "test", "V-domain")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        #后边可以再加上，guo
        # if meet_checkpoint_freq or last_epoch:  # 第一轮跳到这里，如果满足检查点要求或者是最后一个epoch，就保存模型
        #     self.save_model(self.epoch, self.output_dir)  # 第二轮就save了因为是最后一个

    @torch.no_grad()
    def test(self, val_loader=None, test_loader_E=None, test_load_V=None, split=None, domain=None):#,epoch=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        #确定是加载测试集还是验证集
        if split is None: #默认是加载测试集
            split = self.cfg.TEST.SPLIT  # split=test

        if split == "val" and val_loader is not None:
            data_loader = val_loader
        else:
            split = "test"  # in case val_loader is None  先走test
            data_loader_E = test_loader_E  # 得到52维度的test数据集，这里面数据是，一共52个256大小的batch
        # 加载test数据集
        print(f"Evaluate on the *{split}*{domain} set")
        NDCG_5_t = 0.0
        NDCG_10_t = 0.0
        NDCG_20_t = 0.0
        NDCG_50_t = 0.0
        HT_5_t = 0.0
        HT_10_t = 0.0
        HT_20_t = 0.0
        HT_50_t = 0.0
        valid_user_t = 0.0
        best_ndcg_5 = 0.0
        best_epoch_5 = 0
        best_ndcg_50 = 0.0
        best_epoch_50 = 0
        batch_idx_list=[]
        bathce_list =[]
        batch_list_V =[]

        klist = [5, 10, 20, 50]
        for batch_idx, batch in enumerate(tqdm(data_loader_E)):  # 这是全部的
            batch_idx_list.append(batch_idx) #E
            bathce_list.append(batch) #E
        for batch_idx, batch in enumerate(tqdm(test_load_V)):  # 这是全部的
            batch_idx_list.append(batch_idx) #E+V
            batch_list_V.append(batch)  #guo V

        for id, (batch_idx, batch, batch_V) in enumerate(tqdm(zip(batch_idx_list, bathce_list, batch_list_V))):  # batch是全部的
        # for batch_idx, batch in enumerate(tqdm(data_loader)):  # 进度条工具
            input, label_E = self.parse_batch_test(batch)  # 经过embedding编码之后(256,77),(256)
            input_V, label_V= self.parse_batch_test(batch_V)
            if domain=="E-domain":
                label = label_E
            else:
                label= label_V
            usernum = len(input)
            NDCG_5 = 0.0
            NDCG_10 = 0.0
            NDCG_20 = 0.0
            NDCG_50 = 0.0
            HT_5 = 0.0
            HT_10 = 0.0
            HT_20 = 0.0
            HT_50 = 0.0
            valid_user = 0.0

            # itemnum = 4000
            # maxlen = 77   # 改start
            valid_user_t += len(label)
            input_E=input
            # log_feats = None

            log_feats_E, log_feats_V, _, _ = self.model_inference(input_E, input_V, label_E, label_V)
            # log_feats_E = log_feats_E.unsqueeze(0)
            if domain=="E-domain":
                log_feats = log_feats_E
            else:
                log_feats = log_feats_V
            # if domain == "E-domain":
            #     log_feats_E, _, _, _ = self.model_inference(seq, None, label, None)
            #     log_feats_E = log_feats_E.unsqueeze(0)
                # log_feats = log_feats_E
            # if domain == "V-domain":
            #     _, log_feats_V, _, _ = self.model_inference(None, seq, None, label)  # 跳转到trainers->coop.py->CustomerClip->forward 200,4000-1,200,4000
            #     # log_feats_V = log_feats_V.unsqueeze(0)
            #     log_feats = log_feats_V
            predictions = log_feats
        # # # 改
        #
            NDCG = []
            HR = []
            predlist = predictions.argsort()
            for k in klist:
                HR.append(0)
                NDCG.append(0)
                templist = predlist[:, -k:]
                i = 0
                while i<len(label):
                    pos = np.argwhere(templist[i].cpu()==label[i].cpu())
                    if len(pos[0])>0:
                        HR[-1]+=1
                        NDCG[-1]+=1/np.log2(int(k-pos[0][0]+1))
                    else:
                        HR[-1]+=0
                        NDCG[-1]+=0
                    i+=1
            NDCG_5_t += NDCG[0]
            HT_5_t += HR[0]
            NDCG_10_t += NDCG[1]
            HT_10_t += HR[1]
            NDCG_20_t += NDCG[2]
            HT_20_t += HR[2]
            NDCG_50_t += NDCG[3]
            HT_50_t += HR[3]

            if valid_user_t % 100 == 0:  # 每100条写一个省略号的点，512的batch_size，5个点周一个batch结束
                # print('.', end="")
                sys.stdout.flush()

        print(
            "\n=>the result of all the samples\n"
            f"* NDCG@5: {(NDCG_5_t/valid_user_t):.4f}\n"
            f"* HR@5: {(HT_5_t/valid_user_t):.4f}\n"
            f"* NDCG@10: {(NDCG_10_t/valid_user_t):.4f}\n"
            f"* HR@10: {(HT_10_t/valid_user_t):.4f}\n"
            f"* NDCG@20: {(NDCG_20_t/valid_user_t):.4f}\n"
            f"* HR@20: {(HT_20_t/valid_user_t):.4f}\n"
            f"* NDCG@50: {(NDCG_50_t / valid_user_t):.4f}\n"
            f"* HR@50: {(HT_50_t / valid_user_t):.4f}"
        )
        return NDCG_10_t/valid_user_t
    def model_inference(self, seq_E, seq_V, label_E, label_V):
        return self.model(seq_E, seq_V, label_E, label_V, "test")

    def parse_batch_test(self, batch):
        input = batch["seq"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]

#没用用
class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """
    #没有执行
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset决定迭代标记或未标记的数据集
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch_domain(self, type):
        self.set_model_mode("train")  # 设置train模式，而不是eval
        losses = MetricMeter()  # 分隔符=“ ”，然后存储一组度量的平均值和当前值，存成dict格式
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if type=='source':
            self.num_batches = len(self.train_loader_x_E)  # 92  这两个域的数量应该一样，如果不一样的话，就通过随机重复的形式补齐
            train_loader = self.train_loader_x_E
        else:
            self.num_batches = len(self.train_loader_x_V)
            train_loader = self.train_loader_x_V
        end = time.time()
        for self.batch_idx, batch in enumerate(train_loader):  #在这里要对V域的值加上新的编码
            data_time.update(time.time() - end)  # 更新小batch的时间
            loss_summary = self.forward_backward_domain(batch,type) #转向ccop.py->forward_backward,里面更新了loss

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0  #如果是batch数量除以100可以除开了，就开始输出一些指标
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_epoch(self):
        self.set_model_mode("train")  # 设置train模式，而不是eval
        losses = MetricMeter()  # 分隔符=“ ”，然后存储一组度量的平均值和当前值，存成dict格式
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x_E)  # 92  这两个域的数量应该一样，如果不一样的话，就通过随机重复的形式补齐

        end = time.time()
        for self.batch_idx, (batch_E, batch_V) in enumerate(zip(self.train_loader_x_E, self.train_loader_x_V)):
            data_time.update(time.time() - end)  # 更新小batch的时间
            loss_summary= self.forward_backward(batch_E, batch_V, "all")  # 转向ccop.py->forward_backward,里面更新了loss
            # loss_summary_t = self.forward_backward(batch_V, "target")

            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            # losses.update(loss_summary_t)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0  # 如果是batch数量除以100可以除开了，就开始输出一些指标
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
