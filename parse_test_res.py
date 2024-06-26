"""
Goal
---
1. Read test results from log.txt files 1.从日志中读取测试结果。txt文件
2. Compute mean and std across different folders (seeds) 2.计算不同文件夹（种子）的平均值和标准差

Usage
---
Assume the output files are saved under output/my_experiment, 假设输出文件保存在output/my_ experience下，
which contains results of different seeds, e.g., 其包含不同种子的结果。，

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory: 从根目录运行以下命令：

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence 如果你想得到95%的置信度，在参数中加上--ci95
interval instead of standard deviation: 间隔而非标准偏差：

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))

# 参数传入metrics和directory，指标和文件位置，指标：{name,accuracy,regex},文件：output/ox_pets/../ctpmiddle/seed1
def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)  # 这里subdirs:['log.txt','log.txt-2022-08-27-13-47-24']...
    # 代表这里读到了日志txt文件

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")  # , fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:  # 这里在一条一条的读配置的参数
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")

    else:
        parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建解析器
    parser.add_argument("directory", type=str, help="path to directory")  # 添加参数 目录的路径
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"  # 计算95%置信区间，如果想这样的话，就在传参的时候在结尾加上ci95
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")  # 仅分析测试日志
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"  # 解析多个实验，这里我们没有用到，如果目录是另一种形式的话就会用到了
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"  # 要提取的关键字，如何使用的呢？
    )
    args = parser.parse_args()  # 使用parse_args()解析添加的参数

    end_signal = "Finish training"  # 结束信号，默认值是”Finish training“
    if args.test_log:  # 如果参数的test_log是ture的话，
        end_signal = "=> result"  # 结束信号变为”=> result“

    main(args, end_signal)  # 进入主函数，参数：args配置参数，结束信号
