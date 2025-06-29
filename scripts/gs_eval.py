import time
import os
import re
import argparse
import subprocess
from collections import defaultdict

from tqdm import tqdm

GS = ["gs", "3dgrt", "3dgut", "dn-splatter"]


class TrieNode:
    def __init__(self):
        self.children = {}
        self.entries = []


def insert(trie, word):
    node = trie
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
        node.entries.append(word)


def collect_groups(node, groups):
    if len(node.entries) > 1 and all(
        len(child.entries) < len(node.entries) for child in node.children.values()
    ):
        groups.append(sorted(set(node.entries)))
    for child in node.children.values():
        collect_groups(child, groups)


def group_by_max_prefix(dirs):
    trie = TrieNode()
    for d in dirs:
        insert(trie, d)
    groups = []
    collect_groups(trie, groups)

    grouped = set(i for g in groups for i in g)
    ungrouped = [d for d in dirs if d not in grouped]
    for d in ungrouped:
        groups.append([d])
    return groups


def mean(lst):
    return sum(lst) / len(lst) if lst else 0


def std(lst):
    if len(lst) < 2:
        return 0
    mean_value = mean(lst)
    return (sum((x - mean_value) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5


def execute(command):
    result = subprocess.run([command], capture_output=True, text=True, shell=True)
    return result


def extract_model_path(output):
    model_path_re = r"Output folder:\s+(\S+)"
    match = re.search(model_path_re, output)
    if match:
        return match.group(1)
    else:
        return None


def extract_metrics(output):
    metrics_re = r"SSIM\s*:\s*([0-9.]+)\s*[\r\n]+PSNR\s*:\s*([0-9.]+)\s*[\r\n]+LPIPS\s*:\s*([0-9.]+)"
    match = re.search(metrics_re, output)
    if match:
        ssim = float(match.group(1))
        psnr = float(match.group(2))
        lpips = float(match.group(3))
        return ssim, psnr, lpips
    else:
        return None


def dirs_by_dataset(dirs):
    grouped = group_by_max_prefix(dirs)
    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GS on several files.")
    parser.add_argument(
        "--gs",
        type=str,
        default="gs",
        help="""Gaussian Splatting method.\n\
                        Options: "gs", "3dgrt", "3dgut", "dn-splatter".\n\
                        Default: "gs"\n\
                        Make sure that the script is in the working directory of the method you want to use.""",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing data to process.",
    )
    args = parser.parse_args()

    time_results = defaultdict(list)
    lpips_results = defaultdict(list)
    psnr_results = defaultdict(list)
    ssim_results = defaultdict(list)
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' does not exist.")
        exit(1)
    if not os.path.isdir(data_dir):
        print(f"[ERROR] '{data_dir}' is not a directory.")
        exit(1)
    if args.gs not in GS:
        print(
            f"[ERROR] Invalid Gaussian Splatting method '{args.gs}'.\n\
                        Options: {GS}"
        )
        exit(1)
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for d in tqdm(dirs):
        if args.gs == "gs":
            start_time = time.time()
            command = f"python train.py -s {os.path.join(data_dir, d)} --eval"
            stdout_train = execute(command)
            if stdout_train.stderr:
                print(
                    f"[ERROR] Error occurred during training for {d}: {stdout_train.stderr}"
                )
                continue
            model_stdout = stdout_train.stdout
            model_path = extract_model_path(model_stdout)
            if not model_path:
                print(f"[ERROR] Could not find model path in training output for {d}.")
                continue
            command = f"python render.py -m {model_path} -s {os.path.join(data_dir, d)}"
            stdout_render = execute(command)
            if stdout_render.stderr:
                print(
                    f"[ERROR] Error occurred during rendering for {d}: {stdout_render.stderr}"
                )
                continue
            command = f"python metrics.py -m {model_path}"
            stdout_metrics = execute(command)
            if stdout_metrics.stderr:
                print(
                    f"[ERROR] Error occurred during metrics calculation for {d}: {stdout_metrics.stderr}"
                )
                continue
            metrics_output = stdout_metrics.stdout
            metrics = extract_metrics(metrics_output)
            if metrics is None:
                print(f"[ERROR] Could not extract metrics for {d}.")
                continue
            ssim, psnr, lpips = metrics
            print(f"SSIM: {ssim}, PSNR: {psnr}, LPIPS: {lpips}")
            ssim_results[d].append(ssim)
            psnr_results[d].append(psnr)
            lpips_results[d].append(lpips)
            time_results[d].append(time.time() - start_time)
        else:
            print(
                f"[ERROR] Gaussian Splatting method '{args.gs}' is not implemented yet."
            )
        print("[INFO]")
        grouped_dirs = dirs_by_dataset(time_results.keys())
        res = ""
        print("[INFO] Grouped directories by dataset:")
        for group in grouped_dirs:
            group_name = "_".join(group)
            tmp = f"[INFO] Summary: {group_name}\n"
            tmp += f"  Mean SSIM: {mean([ssim_results[d][0] for d in group]):.4f}"
            tmp += f"  Std SSIM: {std([ssim_results[d][0] for d in group]):.4f}\n"
            tmp += f"  Mean PSNR: {mean([psnr_results[d][0] for d in group]):.4f}"
            tmp += f"  Std PSNR: {std([psnr_results[d][0] for d in group]):.4f}\n"
            tmp += f"  Mean LPIPS: {mean([lpips_results[d][0] for d in group]):.4f}"
            tmp += f"  Std LPIPS: {std([lpips_results[d][0] for d in group]):.4f}\n"
            tmp += (
                f"  Mean Time: {mean([time_results[d][0] for d in group]):.2f} seconds"
            )
            tmp += f"  Std Time: {std([time_results[d][0] for d in group]):.2f} seconds"
            res += tmp
            res += "\n"
            print(tmp)
        with open("gs_eval_results.txt", "w") as f:
            f.write(res)
    print("[INFO] Evaluation completed. Results saved to 'gs_eval_results.txt'.")
