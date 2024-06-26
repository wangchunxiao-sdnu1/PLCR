import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image
import numpy as np
from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform


def build_data_loader(
    cfg,
    domain,
    class_E,
    sampler_type="SequentialSampler",
    # sampler_type="RandomSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, domain, class_E, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)  #转到OxfordPets.py-->init,还没有补0

        # Build transform
        if custom_tfm_train is None:  # 这里是对图片的Transform操作，翻转之类的
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x_E = build_data_loader(
            cfg,
            "source",
            dataset.num_classes_E,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,  # 抽样器类型
            data_source=dataset.train_x_E,  # 数据源
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  # batch_size
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,  # 域
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,  #
            tfm=tfm_train,  # build_transform。里边有内容
            is_train=True,
            dataset_wrapper=dataset_wrapper  # 数据封装
        )
        train_loader_x_V = build_data_loader(
            cfg,
            "target",
            dataset.num_classes_E,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,  # 抽样器类型
            data_source=dataset.train_x_V,  # 数据源
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  # batch_size
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,  # 域
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,  #
            tfm=tfm_train,  # build_transform。里边有内容
            is_train=True,
            dataset_wrapper=dataset_wrapper  # 数据封装
        )

        # Build train_loader_u,没有用到
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader_E = None
        val_loader_V = None
        if dataset.val_E:
            val_loader_E = build_data_loader(
                cfg,
                "source",
                dataset.num_classes_E,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val_E,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )
        if dataset.val_V:
            val_loader_V = build_data_loader(
                cfg,
                "target",
                dataset.num_classes_E,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val_V,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader_E = build_data_loader(
            cfg,
            "source",
            dataset.num_classes_E,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_E,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,  # 256
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        test_loader_V = build_data_loader(
            cfg,
            "target",
            dataset.num_classes_E,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_V,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,  # 256
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes_E = dataset.num_classes_E
        self._num_classes_V = dataset.num_classes_V
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname_E = dataset.lab2cname_E
        self._lab2cname_V = dataset.lab2cname_V

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x_E = train_loader_x_E
        self.train_loader_x_V = train_loader_x_V
        self.train_loader_u = train_loader_u
        self.val_loader_E = val_loader_E
        self.val_loader_V = val_loader_V
        self.test_loader_E = test_loader_E
        self.test_loader_V = test_loader_V

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes_E(self):
        return self._num_classes_E

    @property
    def num_classes_V(self):
        return self._num_classes_V

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname_E(self):
        return self._lab2cname_E

    @property
    def lab2cname_V(self):
        return self._lab2cname_V

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes_E", f"{self.num_classes_E:,}"])
        table.append(["# classes_V", f"{self.num_classes_V:,}"])
        table.append(["# train_x_E", f"{len(self.dataset.train_x_E):,}"])
        table.append(["# train_x_V", f"{len(self.dataset.train_x_V):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val_E:
            table.append(["# val_E", f"{len(self.dataset.val_E):,}"])
        if self.dataset.val_V:
            table.append(["# val_V", f"{len(self.dataset.val_V):,}"])
        table.append(["# test_E", f"{len(self.dataset.test_E):,}"])
        table.append(["# test_V", f"{len(self.dataset.test_V):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):  # 数据封装

    def __init__(self, cfg, data_source, domain, class_E, transform=None, is_train=False):
        self.cfg = cfg
        self.domain = domain  #new guo
        self.class_E = class_E,
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input 仅允许在训练期间将图像放大K>1次
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training 仅允许在训练期间将图像放大K>1次
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        # if self.k_tfm > 1 and transform is None:
        #     raise ValueError(
        #         "Cannot augment the image {} times "
        #         "because transform is None".format(self.k_tfm)
        #     )
        #
        # # Build transform that doesn't apply any data augmentation
        # interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        # to_tensor = []
        # to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        # to_tensor += [T.ToTensor()]
        # if "normalize" in cfg.INPUT.TRANSFORMS:
        #     normalize = T.Normalize(
        #         mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        #     )
        #     to_tensor += [normalize]
        # self.to_tensor = T.Compose(to_tensor)
    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "session": item.session,
            "index": idx
        }
        # 20220914guo
        maxlen = self.cfg.TRANSFORMER.MAX_LEN #77  # 这个是指，补齐的长度，就是每条session要有几个item
        seq = np.zeros([maxlen], dtype=np.int32)
        items = item.session.split(',')
        items = list(map(int, items))  #将字符转成int型

        len_s = len(items) if len(items) < maxlen else maxlen  # 前面补零，共50长度
        id_i =len_s-1
        id_s = maxlen-1
        if self.domain=="target":
            for i in range(len_s):
                seq[id_s - i] = items[id_i - i] + self.class_E[0]  #修改的item的编号
        else:
            for i in range(len_s):
                seq[id_s-i] = items[id_i-i]
            # id_s -=1
            # id_i -=1
            # if id_i == -1 or id_s ==-1 : break
        # seq = [i for i in items]
        output["seq"] = torch.LongTensor(seq)
        '''20220913wang
        img0 = read_image(item.impath)
        
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)  # _build_transform_train图片处理
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation
        '''
        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
