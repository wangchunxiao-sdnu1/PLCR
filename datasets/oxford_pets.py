import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()  # 下面是注册的数据类
class OxfordPets(DatasetBase):

    dataset_dir = "Hvideo"

    def __init__(self, cfg):
        # 20220912
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        # self.dataset_dir = os.path.join(root, 'u1.base')
        # self.dataset_dir = os.path.join(root, self.dataset_dir)  # dataset_dir: '/home/wangchunxiao/CoOp/DATA/oxford_pets'
        #
        #
        #
        # if os.path.exists(self.dataset_dir):
        #     train, val, test = self.read_split(self.dataset_dir)  # 把操作文件+图片文件传入这个函数，开始分出train\val\test集
        # 20220913修改代码
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))  # root: '/home/wangchunxiao/CoOp/DATA'
        self.dataset_dir = os.path.join(root, self.dataset_dir)  # dataset_dir: '/home/wangchunxiao/CoOp/DATA/oxford_pets'
        # self.image_dir = os.path.join(self.dataset_dir, "images")  # dataset_dir:'/home/wangchunxiao/CoOp/DATA/oxford_pets/images'
        # self.session_dir = os.path.join(self.image_dir, "images.txt")  # 数据位置
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")  # dataset_dir:'/home/wangchunxiao/CoOp/DATA/oxford_pets/images/annotations'

        # self.split_path_E = os.path.join(self.dataset_dir, "split_hvideo_E.json")  # dataset_dir:'/home/wangchunxiao/CoOp/DATA/oxford_pets/images/split_zhou_OxfordPets.json'
        # self.split_path_V = os.path.join(self.dataset_dir, "split_hvideo_V.json")

        # self.split_path_E = os.path.join(self.dataset_dir, "split_hvideo_E10-26_10.json")  # dataset_dir:'/home/wangchunxiao/CoOp/DATA/oxford_pets/images/split_zhou_OxfordPets.json'
        # self.split_path_V = os.path.join(self.dataset_dir, "split_hvideo_V10-26_10.json")  # movie-book

        self.split_path_E = os.path.join(self.dataset_dir,"split_Amazon_F.json")  # dataset_dir:'/home/wangchunxiao/CoOp/DATA/oxford_pets/images/split_zhou_OxfordPets.json'
        self.split_path_V = os.path.join(self.dataset_dir, "split_Amazon_K.json")

        # self.split_path_E = os.path.join(self.dataset_dir, "TKDEMovie.json")  # 2024-4-27 1
        # self.split_path_V = os.path.join(self.dataset_dir, "TKDEBook.json")

        # self.split_path_E = os.path.join(self.dataset_dir, "TKDEMovie.json")  # 2024-4-27 2
        # self.split_path_V = os.path.join(self.dataset_dir, "TKDEMovielens.json")

        # self.split_path_E = os.path.join(self.dataset_dir, "TKDEBook.json")  # 2024-4-27 3
        # self.split_path_V = os.path.join(self.dataset_dir, "TKDEMovielens.json")

        # self.split_path_E = os.path.join(self.dataset_dir, "split_Amazon_beauty.json")
        # self.split_path_V = os.path.join(self.dataset_dir, "split_Amazon_sports.json")

        # self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")  # dataset_dir: '/home/wangchunxiao/CoOp/DATA/oxford_pets/split_fewshot'
        # mkdir_if_missing(self.split_fewshot_dir)
        # print("the name of dataset:",self.split_path_E,self.split_path_V)
        train_E, val_E, test_E = self.read_split(self.split_path_E)  # 把操作文件+图片文件传入这个函数，开始分出train\val\test集
        train_V, val_V, test_V = self.read_split(self.split_path_V)
        # if os.path.exists(self.split_path):
        #     train, val, test = self.read_split(self.split_path, self.session_dir)  # 把操作文件+图片文件传入这个函数，开始分出train\val\test集
        # else:
        #     trainval = self.read_data(split_file="trainval.txt")
        #     test = self.read_data(split_file="test.txt")
        #     train, val = self.split_trainval(trainval)
        #     self.save_split(train, val, test, self.split_path, self.image_dir)

        # 这里不用
        # num_shots = cfg.DATASET.NUM_SHOTS  # num_shots=-1
        # if num_shots >= 1:  # 没走
        #     seed = cfg.SEED
        #     preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
        #
        #     if os.path.exists(preprocessed):
        #         print(f"Loading preprocessed few-shot data from {preprocessed}")
        #         with open(preprocessed, "rb") as file:
        #             data = pickle.load(file)
        #             train, val = data["train"], data["val"]
        #     else:
        #         train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        #         val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
        #         data = {"train": train, "val": val}
        #         print(f"Saving preprocessed few-shot data to {preprocessed}")
        #         with open(preprocessed, "wb") as file:
        #             pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES  # all
        train_E, val_E, test_E = self.subsample_classes(train_E, val_E, test_E, subsample=subsample)
        train_V, val_V, test_V = self.subsample_classes(train_V, val_V, test_V, subsample=subsample)

        super().__init__(train_x_E=train_E, train_x_V=train_V,
                         val_E=val_E, val_V= val_V,
                         test_E=test_E, test_V= test_V)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)  # 形成路径
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)  # 把这三个添加到item中
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):  # 将train val分开
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath):  # DATA/oxford_pets/images/split_zhou_OxfordPets.json;DATA/oxford_pets/images
        def _convert(items):  # 这一步先不走，在走过split之后，会在train、val、test的时候调用
            out = []  # item是对josn的读取，包括如下内容['Abyssinian_122.jpg', 0, 'abyssinian']
            for session, label, classname in items:  # 这样的话：impath=Abyssinian_122.jpg,label=0,classname=abyssinian
                # impath = os.path.join(path_prefix, impath)  # wang path=image_dir(DATA/oxford_pets/images)+impath=Abyssinian_122.jpg

                item = Datum(session=session, label=int(label), classname=classname)  # itmem就变成了一个对象<dassl.data.datasets.base_dataset.Datum object at 0x7f2a21f5bc40>指向内存中的一个位置
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)  # 这里有train/val/test的读josn的读取划分，划分内容如下
        train = _convert(split["train"])  # 按数据文件划分],
        val = _convert(split["val"])  #
        test = _convert(split["test"])  #

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group  把类分为两组，第一组表示基础类，第二组代表一个新类
        represents base classes while the second group represents
        new classes.


        Args:
            args: a list of datasets, e.g. train, val and test. args：数据集列表，例如train、val和test。
            subsample (str): what classes to subsample.  subsample（str）：要对哪些类进行子采样。
        """
        assert subsample in ["all", "base", "new"]  # subsample=all，断言subsample是否在[]中，如果不在就触发

        if subsample == "all":  # 有放回随机抽样
            return args  # 这里应该是存储了图片数据
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    session=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
