import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum)  # 原来usernum+1
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum )  # 原来是usernum+1

        seq = np.zeros([maxlen], dtype=np.int32)  # seq=maxlen长度的0
        pos = np.zeros([maxlen], dtype=np.int32)  # pos=maxlen长度的0
        neg = np.zeros([maxlen], dtype=np.int32)  # maxlen=200
        nxt = user_train[user][-1]  # 选的那个随机用户的倒数第一个item，根据上例就是user_train[651][-1]=997  nxt:897
        idx = maxlen - 1  # idx=199，减去了最后一个

        ts = set(user_train[user])  # 建立无序不重复的集和，将随机选的那个用户，651中的项目进行无序不重复输出
        for i in reversed(user_train[user][:-1]):  # 这里的i代表的是用户的item_id.返回给定序列值的反向迭代器，这里是user_train[651]中item的正常顺序的倒序，并且把最后那一个数没选，也就是nxt
            seq[idx] = i  # 794，也就是读入的数据，idx也是倒着来的，从199开始，user_train的读数也是倒着来的，从倒数第二个开始
            pos[idx] = nxt  # 最后那个数
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  # 负例采样？无序随机，从1到item_num随机采样，且这个值不能在ts中。。。就是说，这个item不在这个用户的交互中
            nxt = i
            idx -= 1  # idx递减，一直填满200的seq、pos、neg长度
            if idx == -1: break

        return (user, seq, pos, neg)  # 返回得到的seq+正例+负例

    np.random.seed(SEED)  # SEED居然等于779610444?
    while True:
        one_batch = []
        for i in range(batch_size):  # batch_size=128
            one_batch.append(sample())  # one_batch中的四个东西：0：用户id;1:seq;2:pos;3:neg

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)  # maxsize=30，返回队列对象={Queue}
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()  # 启动子进程

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):  # fname:'ml-1m'
    usernum = 0  # 目前都是空的，往里输数据
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')  # 读入数据
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)  # 用户u的list中开始将item往里读

    for user in User:  #
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]  # 从0-倒数第三个是train
            user_valid[user] = []
            user_valid[user].append(User[user][-2])  # 倒数第二个是valid
            user_test[user] = []
            user_test[user].append(User[user][-1])  # 倒数第一个是test
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    # [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    seqs, labels, usernum, itemnum = dataset

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # if usernum>10000:
    #     users = random.sample(range(1, usernum + 1), 10000)
    # else:
    users = range(1, usernum)
    for u in users:

        if len(seqs[u]) < 1 or len(labels[u])<1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # seq[idx] = labels[u][0]#valid[u][0]  #把valid中的item放入序列中
        # idx -= 1
        for i in reversed(seqs[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(seqs[u])
        rated.add(0)
        #把直值放入item
        item_idx = [labels[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()  #为什么排两次

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
#这是专门用于验证集，测试集另一个
def evaluate_valid(model, dataset, args):
    # [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    train, label, usernum, itemnum = dataset

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        # item_idx = [valid[u][0]]
        item_idx=[]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user