from numpy.ma import copy

from dassl.data.datasets import base_dataset


def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


# def compute_dncg(output, label):
#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0
#     for log in output:  # 求取每一条中的u-i分数 mo(256,4000),log(4000)
#         predictions = log  # log本身就是一维，不用变
#         # （1，101）→（101）
#         rank = predictions.argsort().argsort()[0].item()  # rank=3377
#         # 对元素从小到大排序，并返回相应元素的下标。再对下标从小到大排序，返回下标最小的下标，利用这个下标取值。也就是得到最小元素的下标（比索引取值精度更高）
#         valid_user += 1  # 0.0->1.0
#
#         if rank < 50:  # rank=2<10就是命中？所以rel(i)=1?
#             NDCG += 1 / np.log2(rank + 2)  # rel(i)=1?
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     self._ndcg = NDCG
#     self._ht = HT
#     self._valid_user = valid_user








# def evaluate_test(train,test,val,args):  # model, dataset, args):  # 应输入output+label ，再加一个test集
#     dataset = [train, test, val, args.batch_size, args.n_cls]
#     [output, label, test]=copy.deepcopy(dataset)
#
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
# # train:6040，valid:6040,test:6040,,这里的test和valid和label是不是有什么关系？其实label是与class相同的。usernum:6040,itemnum:3416
#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0
#
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)  # 用户数目小于1万，就从1-usernum+1中顺序输出
#     for u in users: # u从1，到6041
#
#         if len(train[u]) < 1 or len(test[u]) < 1: continue  # 两个都不小于1
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)  # maxlen=200
#         idx = args.maxlen - 1  # idx=199
#         seq[idx] = valid[u][0]  # seq[最后一个]=valid[1][0]，从后往前填数，倒数第二，倒数第三，逐个对应着填入train的值
#         idx -= 1
#         for i in reversed(train[u]):  # 遍历逆序序列
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])  # set建立一个无序不重复数据集长度为train[u]的长度
#         rated.add(0)  # 加入一个0进去
#         item_idx = [test[u][0]]  # 一个单独的数
#         for _ in range(100):  # -：99
#             t = np.random.randint(1, itemnum + 1)  # 包括low，但是不包含high,在其中生成随机数，t=699,也就是随机取一个项目id
#             while t in rated: t = np.random.randint(1, itemnum + 1)  # 如果取到的这个项目id,在加了0的u用户的item中，t就再随机取值，也就是不取train中的item,也不取0
#             item_idx.append(t)  # 直到它不存在在用户的item中，再将其添加到item_idx中
#             # item_idx的组成：第一个是test[u]的值，后边100个数都必须不是0或者train[u]中的item id,
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])  # 这里是传了三个值，第一个是当前uid,第二个是126行seq的值，第三个是0-100，共101个item测试序列
#         predictions = predictions[0] # - for 1st argsort DESC 将二维变一维
#         # （1，101）→（101）
#         rank = predictions.argsort().argsort()[0].item()
#         # 对元素从小到大排序，并返回相应元素的下标。再对下标从小到大排序，返回下标最小的下标，利用这个下标取值。也就是得到最小元素的下标（比索引取值精度更高）
#         valid_user += 1
#
#         if rank < 10:  # rank=2<10就是命中？所以rel(i)=1?
#             NDCG += 1 / np.log2(rank + 2)  # rel(i)=1?
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user
#
# def evaluate(model, dataset, args):  # 应输入output+label
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
# # train:6040，valid:6040,test:6040,,这里的test和valid和label是不是有什么关系？其实label是与class相同的。usernum:6040,itemnum:3416
#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0
#
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)  # 用户数目小于1万，就从1-usernum+1中顺序输出
#     for u in users: # u从1，到6041
#
#         if len(train[u]) < 1 or len(test[u]) < 1: continue  # 两个都不小于1
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)  # maxlen=200
#         idx = args.maxlen - 1  # idx=199
#         seq[idx] = valid[u][0]  # seq[最后一个]=valid[1][0]，从后往前填数，倒数第二，倒数第三，逐个对应着填入train的值
#         idx -= 1
#         for i in reversed(train[u]):  # 遍历逆序序列
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])  # set建立一个无序不重复数据集长度为train[u]的长度
#         rated.add(0)  # 加入一个0进去
#         item_idx = [test[u][0]]  # 一个单独的数
#         for _ in range(100):  # -：99
#             t = np.random.randint(1, itemnum + 1)  # 包括low，但是不包含high,在其中生成随机数，t=699,也就是随机取一个项目id
#             while t in rated: t = np.random.randint(1, itemnum + 1)  # 如果取到的这个项目id,在加了0的u用户的item中，t就再随机取值，也就是不取train中的item,也不取0
#             item_idx.append(t)  # 直到它不存在在用户的item中，再将其添加到item_idx中
#             # item_idx的组成：第一个是test[u]的值，后边100个数都必须不是0或者train[u]中的item id,
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])  # 这里是传了三个值，第一个是当前uid,第二个是126行seq的值，第三个是0-100，共101个item测试序列
#         predictions = predictions[0] # - for 1st argsort DESC 将二维变一维
#         # （1，101）→（101）
#         rank = predictions.argsort().argsort()[0].item()
#         # 对元素从小到大排序，并返回相应元素的下标。再对下标从小到大排序，返回下标最小的下标，利用这个下标取值。也就是得到最小元素的下标（比索引取值精度更高）
#         valid_user += 1
#
#         if rank < 10:  # rank=2<10就是命中？所以rel(i)=1?
#             NDCG += 1 / np.log2(rank + 2)  # rel(i)=1?
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user
#
#
# # evaluate on val set
# def evaluate_valid(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0].item()
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user