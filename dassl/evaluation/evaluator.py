import collections
import copy
import math
import sys
from random import random

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from numpy import array
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        # self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._ndcg = []
        self._hr = []
        self._recall = []
        # if cfg.TEST.PER_CLASS_RESULT:
        #     assert lab2cname is not None
        #     self._per_class_res = defaultdict(list)

    # def model_inference(self, input):
    #     return self.model(input)
    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._ndcg20 = []
        self._hr20 = []
        self._ndcg50 = []
        self._hr50 = []
        self._recall = []
        if self._per_class_res is not None:  # 等于空
            self._per_class_res = defaultdict(list)

    def process(self, NDCG_20, HR_20, NDCG_50, HR_50):
        self._ndcg20 = NDCG_20
        self._hr20 = HR_20
        self._ndcg50 = NDCG_50
        self._hr50 = HR_50

    def evaluate(self):
        results = OrderedDict()  # 记忆插入顺序的字典
        NDCG_20 = self._ndcg20*100
        NDCG_50 = self._ndcg50*100
        HR_20 = self._hr20 * 100
        HR_50 = self._hr50*100
        Recall = self._recall

        # err = 100.0 - acc
        # macro_f1 = 100.0 * f1_score(
        #     self._y_true,
            # self._y_pred,
            # average="macro",
            # labels=np.unique(self._y_true)   # 去除数组中的重复数字，并进行排序之后输出
        # )

        # The first value will be returned by trainer.test()
        results["NDCG_20"] = NDCG_20
        results["HR_20"] = HR_20
        results["NDCG_50"] = NDCG_50
        results["HR_50"] = HR_50
        # results["Recall"] = Recall
        # results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* ndcg@20: {NDCG_20:.1f}%\n"
            f"* ht@20: {HR_20:.1f}%\n"
            f"* ndcg@50: {NDCG_50:.1f}%\n"
            f"* ht@50: {HR_50:.1f}%"
        )

        if self._per_class_res is not None:  # None
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                # classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    # f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:  # False
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
