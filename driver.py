"""Runs the whole study: trains every combination of settings, then tests every saved model.

    python driver.py train          train every run (finished runs are skipped)
    python driver.py test           test every saved model and print the results
    python driver.py test --log     ...and also save them to results/
    python driver.py plots          draw the charts from results/test_results.csv

Add --quick to any of them for a tiny version in quick/ that checks everything works and
measures speed. Delete quick/ afterwards.
"""
import argparse
import glob
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from multiprocessing import get_context

import pandas as pd
import torch

import testing
import trainer

ALGOS = ["ppo", "sac"]
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
SIZES = [[16, 16], [64, 64], [256, 256]]
SEEDS = [0, 1, 2]
STAGES = [125, 250, 500, 1000, 2000, 4000]  # training episodes at which the weights are saved

QUICK = dict(learning_rates=[3e-4, 1e-3], sizes=[[16, 16], [256, 256]], seeds=[0], stages=[10, 30],
             root=os.path.join(trainer.ROOT, "quick"))
QUICK_ENVS_PER_CONDITION = 2

GPU_WORKERS = 2  # big SAC runs at once on the RTX 3050


def train_job(run, stages, root, device):
    torch.set_num_threads(1)  # one CPU thread per run, since many runs share the CPU
    return trainer.train(*run, stages=stages, device=device, root=root)


def test_job(model_path, root):
    torch.set_num_threads(1)
    return testing.test(model_path, root=root)


def stop_pool(pool):
    """Ctrl+C: stop the worker processes straight away instead of waiting for their runs to end."""
    for process in list((getattr(pool, "_processes", None) or {}).values()):
        process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


def train_all(algos=ALGOS, learning_rates=LEARNING_RATES, sizes=SIZES, seeds=SEEDS, stages=STAGES,
              root=trainer.ROOT, cpu_workers=None, gpu_workers=None):
    """Trains every algo x lr x size x seed run, several at once: big SAC runs on the GPU, the rest on the CPU."""
    runs = [(algo, lr, size, seed) for algo in algos for lr in learning_rates for size in sizes for seed in seeds]
    todo = [run for run in runs if not trainer.finished(*run, stages, root)]
    gpu_workers = (GPU_WORKERS if gpu_workers is None else gpu_workers) if torch.cuda.is_available() else 0
    cpu_workers = max(1, cpu_workers or (os.cpu_count() or 2) - gpu_workers - 1)
    print(f"{len(runs)} runs: {len(runs) - len(todo)} already trained, {len(todo)} to go "
          f"({cpu_workers} at once on the CPU, {gpu_workers} on the GPU)", flush=True)
    if not todo:
        return
    todo.sort(key=lambda run: (run[0] != "sac", -max(run[2])))  # longest runs first
    gpu_queue = [run for run in todo if gpu_workers and trainer.pick_device(run[0], run[2]) == "cuda"]
    cpu_queue = [run for run in todo if run not in gpu_queue]

    spawn = get_context("spawn")
    pools = {"cpu": ProcessPoolExecutor(cpu_workers, mp_context=spawn)}
    if gpu_workers:
        pools["cuda"] = ProcessPoolExecutor(gpu_workers, mp_context=spawn)
    running, failed = {}, []
    try:
        while cpu_queue or gpu_queue or running:
            busy = Counter(device for _, device in running.values())
            while gpu_queue and busy["cuda"] < gpu_workers:
                run = gpu_queue.pop(0)
                running[pools["cuda"].submit(train_job, run, stages, root, "cuda")] = (run, "cuda")
                busy["cuda"] += 1
            # Once the CPU has nothing else to do, it helps with big SAC runs, but only while the GPU still
            # has more than two rounds of them waiting, so a slow CPU run never holds up the end
            while busy["cpu"] < cpu_workers and (cpu_queue or len(gpu_queue) > 2 * gpu_workers):
                run = cpu_queue.pop(0) if cpu_queue else gpu_queue.pop()
                running[pools["cpu"].submit(train_job, run, stages, root, "cpu")] = (run, "cpu")
                busy["cpu"] += 1
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                run, device = running.pop(future)
                try:
                    future.result()
                except Exception as error:
                    failed.append(run)
                    print(f"{trainer.model_name(*run)}  FAILED on {device}: {error!r}", flush=True)
    except KeyboardInterrupt:
        print("Stopping. Finished runs are kept; unfinished runs start over next time.", flush=True)
        for pool in pools.values():
            stop_pool(pool)
        raise SystemExit(1)
    for pool in pools.values():
        pool.shutdown()
    print(f"training finished, {len(failed)} runs failed" + (" (run train again to retry them)" if failed else ""))


def test_all(log=False, root=trainer.ROOT, envs_per_condition=testing.ENVS_PER_CONDITION, workers=None):
    """Tests every saved model on the test planets and prints each model's table.
    log=True also saves the landings to results/, skipping models that are already saved there."""
    models = sorted(glob.glob(os.path.join(root, "resources", "*", "*_ep*.zip")))
    if not glob.glob(os.path.join(root, "resources", "envs", "*.json")):
        testing.make_envs(envs_per_condition, root)
    results_file = os.path.join(root, "results", "test_results.csv")
    if log and os.path.exists(results_file):
        logged = set(pd.read_csv(results_file, usecols=["model"])["model"])
        models = [path for path in models if testing.model_info(path)["model"] not in logged]
        print(f"{len(logged)} models already in {results_file}", flush=True)
    print(f"testing {len(models)} models", flush=True)
    if not models:
        return
    pool = ProcessPoolExecutor(max(1, workers or (os.cpu_count() or 2) - 1), mp_context=get_context("spawn"))
    try:
        futures = {pool.submit(test_job, path, root): path for path in models}
        for future in as_completed(futures):
            try:
                results = future.result()
            except Exception as error:
                print(f"{os.path.basename(futures[future])}  FAILED: {error!r}", flush=True)
                continue
            if log:
                testing.save(results, root)
    except KeyboardInterrupt:
        print("Stopping. Models tested so far are saved." if log else "Stopping.", flush=True)
        stop_pool(pool)
        raise SystemExit(1)
    pool.shutdown()
    if log:
        testing.summarize(root)
        print(f"saved {results_file} and summary.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["train", "test", "plots"])
    parser.add_argument("--log", action="store_true", help="test: also save the results to results/")
    parser.add_argument("--quick", action="store_true", help="tiny version in quick/ to check everything works")
    parser.add_argument("--cpu-workers", type=int, help="runs at once on the CPU (default: all threads but one)")
    parser.add_argument("--gpu-workers", type=int, help=f"runs at once on the GPU (default {GPU_WORKERS})")
    args = parser.parse_args()
    root = QUICK["root"] if args.quick else trainer.ROOT
    if args.task == "train":
        train_all(**(QUICK if args.quick else {}), cpu_workers=args.cpu_workers, gpu_workers=args.gpu_workers)
    elif args.task == "test":
        per_condition = QUICK_ENVS_PER_CONDITION if args.quick else testing.ENVS_PER_CONDITION
        test_all(args.log, root, per_condition, args.cpu_workers)
    else:
        import plots  # only here, so the training and testing workers don't load matplotlib

        plots.make_plots(root)
