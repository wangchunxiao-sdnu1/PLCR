import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from clip.model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# if torch.__version__.split(".") < ["1", "7", "1"]:
#     warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {  # 字典{key:value}
    "RN50": "SASRec/ml-1m_default/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

## "SASRec/ml-1m_default/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth",
def _download(url: str, root: str = os.path.expanduser("~/sasr/dowmload")):  # root='/home/wangchunxiao/sasr/download'   下载模型，CLIP-RN50。当前用户的目录：在clip下建立了一个__pycache__里边有四个文件
    os.makedirs(root, exist_ok=True)  # 创建一个目录，exist_ok=False就会报错，=True不会报错
    filename = os.path.basename(url)  # 取路径的basename，也就是尾巴  filename:'SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth'

    expected_sha256 = url.split("/")[-2]   # expected_sha256:'ml-1m_default'
    download_target = os.path.join(root, filename)  # root:home/.cache/clip;filename=引用链接的basename

    # download_target:'/home/wangchunxiao/sasr/dowmload/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth'
    # if os.path.exists(download_target) and not os.path.isfile(download_target):  # 如果download_target存在且不为文件
    #     raise RuntimeError(f"{download_target} exists and is not a regular file")  # 返回runtime error不是常规文件
    #
    # if os.path.isfile(download_target):  # 如果download是文件，并符合下边这个式子，返回download_target
    #     if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:  # 返回摘要，作为十六进制数据字符串值，加密解密的用法
    #         return download_target
    #     else:
    #         warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
    #
    # with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
    #     # 这模块用于向浏览器请求download_target； download文件作为output
    #     with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:  # 实现可视化进度
    #         # 返回source的Content-Length对应的key值作为loop
    #         while True:
    #             buffer = source.read(8192)
    #             if not buffer:
    #                 break
    #
    #             output.write(buffer)
    #             loop.update(len(buffer))
    #
    # if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
    #     raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target  # 44行


def _transform(n_px):  # torchvision.transforms主要是用于常见的一些图形变换。

    return Compose([  # 是Transformers()下的函数，用于串联多个图片变换的操作，
        # 参数是一个列表，这个列表中的元素就是想要执行的transformer操作
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:  # 将有用的模型名字列一个列表
    """Returns the names of available CLIP models"""  # 返回可用CLIP模型的名字
    return list(_MODELS.keys())  # 返回上面提到的模型的keys，也就是名字，MODELS={key,value}


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=False):
    """Load a CLIP model  加载一个CLIP模型

    Parameters
    ----------
    name : str 由‘clip.available_models()’列出的模型名称，或者，包含state_dict的模型checkpoint的路径
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device] 设备：union[],用于加载模型的设备
        The device to put the loaded model

    jit : bool 是加载优化的JIT模型，还是加载更易破解的非JIT模型（默认）
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    Returns
    -------
    model : torch.nn.Module # 返回的结果，模型是torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor] 预处理：Callable[[PIL.Image],torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        将PIL图像转换为张量的torchvision变换，返回的模型可以将其作为输入
    """
    if name in _MODELS:  # 按照名字download模型
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive加载JIT存档
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names 修补设备名称
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)  # 将模型应用到设备上
    patch_device(model.encode_image)  # 将model.py中的encoder_image进行patch_sevice
    patch_device(model.encode_text)   # 将model.py中的encoder_text进行patch_sevice

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())  # 返回，1、上文操作完成的模型+2、将上文提到的model，使用mode.py中的input_resolution函数，得到。。还不知道


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s) 返回给定输入字符串的标记化表示

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize 要标记的输入字符串或输入字符串列表

    context_length : int
        The context length to use; all CLIP models use 77 as the context length 要使用的上下文长度；所有剪辑模型都使用77作为上下文长度

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length 如果文本编码长于上下文长度，是否截断文本

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    包含结果标记的二维张量，shape=[输入字符串的数量，context_length]
    """
    if isinstance(texts, str):
        texts = [texts]  # text是要标记的输入字符串或输入字符串列表

    sot_token = _tokenizer.encoder["<|startoftext|>"]  # encoder是simpler_tokenizer里边的,sot代表start of text
    eot_token = _tokenizer.encoder["<|endoftext|>"]  # eot代表end of text
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]  # 所有token的格式=开始标志+编码后的text+遍历texts中的[eto_token]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)  # result是一个维度为[所有token长度，上下文长度=77]的0矩阵
    # 为什么_tokenizer.encode(text)是长度=20的？类名不是长度只有1吗？这样的话类名长度是20啊。
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:  # 判断如果token超过了77，要截断
            if truncate:
                tokens = tokens[:context_length]  # token是0-76+eot_token
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
        # 把result的原始0矩阵填充，每一行都是token的tensor形式,变成[49406, 343, 343343, , 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 16342, 690, 36638, 269, 49407,0,0,0...]一共77个，剩下的都用0填充

    return result  # 结果是start+初始化词+end
