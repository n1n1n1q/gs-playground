import time
import argparse
import subprocess

def mean(lst):
    return sum(lst) / len(lst) if lst else 0

def std(lst):
    if len(lst) < 2:
        return 0
    mean_value = mean(lst)
    return (sum((x - mean_value) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

def execute(command):
    result = subprocess.run([command], capture_output=True, text=True, shell=True)
    if result.stderr:
        print("[INFO] Error occured during command execution")
        print(result.stderr)
        print("[INFO] Command that failed:")
        print(command)
    return not bool(result.stderr) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a command repeatedly.")
    parser.add_argument("command", help="The command to run.")
    parser.add_argument("data_dirs", nargs="*", help="Directories containing data to process.")
    parser.add_argument("--times", type=int, default=1, help="Number of times to run the command.")
    args = parser.parse_args()

    results = {}
    for data_dir in args.data_dirs:
        for i in range(args.times):
            start_time = time.time()
            command = f"{args.command} {data_dir}"
            print(f"Running command: {command} (Run {i + 1}/{args.times})")
            if not execute(command):
                i -= 1
                continue
            end_time = time.time()
            elapsed_time = end_time - start_time
            results[data_dir] = results.get(data_dir, []) + [elapsed_time]
            print(f"[INFO] Elapsed time for run {i + 1}: {elapsed_time:.2f} seconds")

    res = ""
    for data_dir, times in results.items():
        tmp = ""
        tmp += f"[INFO] Summary for {data_dir}:\n"
        tmp += f"[INFO]   Mean elapsed time: {mean(times):.2f} seconds\n"
        tmp += f"[INFO]   Standard deviation: {std(times):.2f} seconds\n"
        tmp += f"[INFO]   All elapsed times: {', '.join(f'{t:.2f}' for t in times)}\n"
        res += tmp
        res += "\n"
        print(tmp)
    with open("timed_run_results.txt", "w") as f:
        f.write(res)
