"""Model names and file paths. Kept out of trainer.py, which loads torch, so `driver.py progress` stays small."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def lr_tag(lr):
    """0.0003 -> '3e-4'"""
    mantissa, exponent = f"{lr:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def model_name(algo, lr, size, seed):
    """e.g. sac_lr3e-4_net256x256_seed0 (the stage files add _ep1000)"""
    return f"{algo}_lr{lr_tag(lr)}_net{'x'.join(map(str, size))}_seed{seed}"


def stage_path(algo, lr, size, seed, episodes, root=ROOT):
    return os.path.join(root, "resources", algo, f"{model_name(algo, lr, size, seed)}_ep{episodes}.zip")


def finished(algo, lr, size, seed, stages, root=ROOT):
    return all(os.path.exists(stage_path(algo, lr, size, seed, stage, root)) for stage in stages)
