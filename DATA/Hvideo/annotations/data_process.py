##定义getdict()函数；以itemE说明调用该函数的处理过程。
import os
import re
import os.path as osp
import errno
import json
import readline


# from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import mkdir_if_missing
def getdict(r_path):  # r_path:形参；
        itemdict = {}  # itemdict:定义一个词典；以itemE为例，第一次循环：{'E82':0};第二次循环：{'E82':0,'E83':1}......
        with open(r_path, 'r') as f:
            items = f.readlines()  # 打开本地文件，模式'r/w/a':只读，只写，追加内容；;读入Elist中的数据, items:['0\tE82\t53091\n','1\tE83\t45339\n','2\tE48\t36598\n,......]
        ## 把本地文件中的item写进itemdict字典中
        for item in items:  # 循环结构，item: ['0\tE82\t53091\n']
            # #这里是用两个函数进行处理，经过这一步后，变为一个列表
            # #.strip()用于移除字符串头尾指定的字符（默认为移除字符串头尾的空格或换行符）.split()默认以空格拆分，对字符串进行切片，经过这一步后变为一个列表；
            item = item.strip().split(
                '\t')  # 两个函数进行处理，得到列表item=['0','E82','53091'];
            itemdict[item[1]] = int(item[0])  # 字典名[键]=值；字典的定义
        return itemdict  # 输出字典itemdict{'E82':0,'E83':1,E48':2,...} 注：E:0-3388共3389项；V:0-16430共16431项；

def getdata(datapath,itemE,itemV, userU):  ## 获得分开的U-E，U-V数据集
    # traindata_sess.txt; item E=dict{'E82':0,'E83':1,...}; item V=dict{'V18':0,"V59':1,...}
    with open(datapath, 'r') as f:
        sessions = []  # 定义一个list
        for line in f.readlines():  # line为str型；line='18865380\tE241\tE232\tE392\tE80\tE81\tE192\tE392\tE3\tE239\tE151\tE28\tE12\tV289\tV5\tV10398\tV496\tV138\tV326\tV232\tV7638\tV5578\tV935\tV3032\tV326\tV7193\tV155\tE392\tV326'
            session = []  # session为list型，session=[218,468,71,11,16,77,71,164,255,12,49,27,3451,3393,3394,3393,6297,3443,3420,3427,3395,6963,7855,5261,7516,3427,5103,3391,71,3427]
            line = line.strip().split('\t')  # 经两个函数处理，得到列表['18865380','E241','E232','E392','E80''E81', 'E192', 'E392', 'E3', 'E239', 'E151', 'E28', 'E12', 'V289', 'V5', 'V74', 'V5', 'V10398', 'V496', 'V138', 'V326', 'V232', 'V7638', 'V5578', 'V935', 'V3032', 'V326', 'V7193', 'V155', 'E392', 'V326']
            user = line[0]
            session.append(userU[user])      # 将用户转换为索引值
            for item in line[1:]:  # 从line中的第二项开始，遍历line中的item，从'E241'到'V326'；for循环将line转换为对应的索引列表；
                item = item.split('|')[0]
                if item in itemE:
                    session.append(itemE[item]) ##在itemE中E241的索引是218
                else:
                    session.append(itemV[item] + len(itemE))  # 为了区分序列中的E与V物品，len(itemE)=3389；
            sessions.append(session)  # 定义一个sessions列表，把所有session放在一起；形式[user,E,V]
    return sessions

def processdata(dataset, item_E):  # dataset为上述的sessions列表（9-18这次代码中session中包括E域和label，不包括V域事情）
    data_E = []
    data_V = []
    seqE = []  # E域中的数据，形式为：user id，item
    seqV = []
    for session in dataset:  # list 一条转换后的30item的序列循环
        temp_E = []  # list
        temp_V = []
        seq1 = []  # list
        seq2 = []  # list
        for item in session[1:-2]:  # session中的前28个item,不加最后两项item，2060中session最后两项618，4323是干什么的

            if item < len(item_E):  # #seq1添加的是E域中的item,相当于[0]
                seq1.append(item)
            else: #添加的V域中的item
                seq2.append(item - len(item_E))  # itemE=3389 #seq2添加的是V域中的item,相当于[1]
        temp_E.append(seq1)
        temp_V.append(seq2)  # V域
        temp_E.append(session[-2])  # E域的label                                处理数据保证这一个就是target_A，输出倒数第二个item  618？？
        temp_V.append(session[-1]-len(item_E))  # V域的label                  处理数据保证这一个就是target_B, 输出倒数第一个item 4323-3389=934？ V
        data_E.append(temp_E)
        data_V.append(temp_V)
        seqE.append(seq1)  ##形式：【用户id】
        seqV.append(seq2)  ##形式：【用户id1，Vitem1，Vitem2，Vitem3】【用户id2，Vitem1，Vitem2，Vitem3】
    return data_E, data_V, seqE, seqV

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))
def save_split(train, val, test, filepath):
        def _extract(items):
            out = []
            for item in items:
                sequence = str(item[0]).replace('[', '').replace(']', '')  # sequence
                label = item[1]
                classname = str(label)  # item.classname
                out.append((sequence, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

def list_txt(path, train, valid, test):
    '''
    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    root = '/home/guolei/pycode/Coop/DATA/HVideo'
    path = os.path.join(root, path)
    # if os.path.exists(path):
    save_split(train, valid, test, path)

# 定义好路径，调用getdict()函数：
path = 'Elist.txt'  # 定义路径path
itemE = getdict(path)  # 调用getdict()函数，得到一个词典itemE{'E82':0,'E83':1,...};
path = 'Vlist.txt'
itemV = getdict(path)  # 调用getdict()函数，得到一个词典itemV{'V18':0,'V59':1,...}；
path = 'userlist.txt'
userU = getdict(path)

# 定义好路径，调用getdata()函数
traindatapath = 'traindata_sess.txt'
validdatapath = 'validdata_sess.txt'
testdatapath = 'testdata_sess.txt'
alldatapath = 'alldata_sess.txt'
traindata = getdata(traindatapath, itemE, itemV, userU=userU)  # 将训练集的数据导进来，得到不同用户的索引序列，每个包含30项item ,traindata_E,traindata_V
validdata = getdata(validdatapath, itemE, itemV, userU=userU)  # 将验证集.......,validdata_E,validdata_V
testdata = getdata(testdatapath, itemE, itemV, userU=userU)  # 将测试集.......,testdata_E,testdata_V
alldata = getdata(alldatapath, itemE, itemV, userU=userU)  # ,alldata_E,alldata_V

trainE, trainV, session_trainE, session_trainV = processdata(traindata, item_E=itemE)  # 将训练集数据进行预处理，形成新的sessions，变成8维的数据
validE, validV, session_validE, session_validV = processdata(validdata, item_E=itemE)  # 将验证集......
testE, testV, session_testE, session_testV = processdata(testdata, item_E=itemE)  # 将测试集......
# allE, allV, session_allE, session_allV = processdata(alldata, item_E=itemE)

list_txt('../split_hvideo_E.json', trainE, validE, testE) #生成三元组
list_txt('../split_hvideo_V.json', trainV, validV, testV)
# list_txt(path='trainV.txt', list=trainV)
# list_txt(path='session_trainE', list=session_trainE)
# list_txt(path='session_trainV', list=session_trainV)

# list_txt(path='traindata_E.txt', list=traindataE)
# list_txt(path='traindata_V.txt', list=traindataV)
# list_txt(path='testdata.txt', list=testdata)
# list_txt(path='testdata_E.txt', list=testdataE)
# list_txt(path='testdata_V.txt', list=testdataV)
# list_txt(path='validdata.txt', list=validdata)
# list_txt(path='validdata_E.txt', list=validdataE)
# list_txt(path='validdata_V.txt', list=validdataV)

