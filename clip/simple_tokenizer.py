import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():  # 取路径+下载的包名称=下载包的路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")  # 取__file__的绝对路径，然后取掉文件名，保留目录，再将.txt.gz与路径拼接起来


@lru_cache()
def bytes_to_unicode():  # 字节到编码的转换，返回对应数字与符号的字典，{数字：符号}
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings. 返回utf-8字节列表和相应的unicode字符串列表。
    The reversible bpe codes work on unicode strings. 可逆bpe代码在unicode字符串上工作。
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs. 可逆bpe代码在unicode字符串上工作。
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.为了避免这种情况，我们需要utf-8字节和unicode字符串之间的查找表。
    And avoids mapping to whitespace/control characters the bpe code barfs on.并避免映射到bpe代码所依赖的空白/控制字符。
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    # ord是取字符的ascii值，range()=range(！的ascii,~的ascii)，list转换为list类型[！的ascii,~的ascii]
    # 三个list相加就变成了，拼接成一整个list
    cs = bs[:]  # cs=[range1,range2,range3]
    n = 0
    for b in range(2**8):  # 2的8次方=256，从0到256
        if b not in bs:
            bs.append(b)  # 将除了之前有过的数字，其他的数字都拼接在后面
            cs.append(2**8+n)  # 如果没在之前的数字里出现过，就以n+256的形式append在后面
            n += 1
    cs = [chr(n) for n in cs]  # 打印数字代变的字符，这里也就是把ASCII码转成字符吧
    return dict(zip(bs, cs))  # （bs中的ASCII，与，cs中ASCII代变的字符）组成字典形式{bs:cs}


def get_pairs(word):  # 得到（0，1）（1，2）（2，3）这样的word pair
    """Return set of symbol pairs in a word.返回单词中的符号对集合
    Word is represented as tuple of symbols (symbols being variable-length strings).单词表示为符号元组（符号为可变长度字符串）
    """
    pairs = set()  # 新的空集对象
    prev_char = word[0]  # prev_char是单词的第一个
    for char in word[1:]:
        pairs.add((prev_char, char))  # pairs是从word的第二个开始的，变成（0，1）（1，2）（2，3）这样的word pair
        prev_char = char
    return pairs


def basic_clean(text):  # 基础清洁
    text = ftfy.fix_text(text)  # 减少文本损坏的模块
    text = html.unescape(html.unescape(text))  # 转换所有命名和数字字符引用（例如，&gt；，&#62；，&x3e；）在字符串s中转换为相应的unicode字符。
    return text.strip()


def whitespace_clean(text):  # 空格清洁
    text = re.sub(r'\s+', ' ', text)  # 将所有的空白字符，比如，’ ‘，/n,/s,/t都转化为空格
    text = text.strip()
    return text
# 处理text中内容，用空格替换\s+
# 返回通过用替换repl替换字符串中模式的最左边（或最右边，使用反向模式）非重叠出现而获得的字符串。
# repl可以是字符串，也可以是可调用的；如果是字符串，则处理其中的反斜杠转义；如果是可调用的，则传递匹配对象，并且必须返回要使用的替换字符串。


class SimpleTokenizer(object):  # 简单分词器的类
    def __init__(self, bpe_path: str = default_bpe()):  # bpe路径=default_bpe,取路径+下载的包名称=下载包的路径
        self.byte_encoder = bytes_to_unicode()  # 字节到编码的转换
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # 将byte_encoder中的{数字：符号}调过来了，变成{符号：数字}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')  # 读取gzip文件，并按\n符号以逗号隔开，‘huhu\nvrge’变成[‘huhu‘,'vrge’]
        merges = merges[1:49152-256-2+1]  # 从第二个开始取，到第几不理解
        merges = [tuple(merge.split()) for merge in merges]  # tuple(..)=merges中的分着的数值,tuple形式是(, , , )
        vocab = list(bytes_to_unicode().values())  # 取字典里的值，形成列表
        vocab = vocab + [v+'</w>' for v in vocab]  # 变成[原vocab , v</w>,v</w>,v</w>, v</w>]
        for merge in merges:
            vocab.append(''.join(merge))  # 相同行索引的拼接到一起，列数增加，行数不变，字节转编码的字典的value{数字+W}，join bpe_path中的tuple形式的内容
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 在末尾添加上start,end
        self.encoder = dict(zip(vocab, range(len(vocab))))  # vocab,与range（）组成[(v,r),(v,r),(v,r)]两两对应的列表，之后变为dict{v:r}
        self.decoder = {v: k for k, v in self.encoder.items()}  # 将其逆转，变成{r:v}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # 将文件中的东西也按上述流程走一遍
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)  # 重新编译，忽略大小写

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)  # token的从头到-1位置，再加-1位置，再加个’</w>‘
        pairs = get_pairs(word)  # 将word中的一项项变为两项拼接的pair，（0，1）（1，2）（2，3）

        if not pairs:  # 如果不是成对的，只返回token+</w>
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:  # 文件中的字典形式
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):  # text是类名时：把类名按照空格解析，比如American bulldog，先解析American的值，得2151，再得bulldog的值：15611
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()  # 去掉空格（包括回车等）
        for token in re.findall(self.pat, text):  # 返回字符串中所有匹配项的列表。
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))  # 将token.encoder中的b填入进byte_encoder[]中去
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens  # 最后返回的值是类名American bulldog的值[2151,15611]，这样的作用是可以解析类型将其，美国是一个标签，bulldog也是一个标签。但是分词准则会让其得到两个码

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
