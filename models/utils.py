import pandas as pd
import os
import shutil


def find_best_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None

    progress_csv = pd.read_csv(os.path.join(output_dir, "training_progress_scores.csv"))
    max_f1_idx = progress_csv['f1'].idxmax()
    row = progress_csv.iloc[[max_f1_idx]]
    step = row["global_step"].values[0]

    for checkpoint in os.listdir(output_dir):
        if f'checkpoint-{step}' in checkpoint:
            return checkpoint

    return None


def clean(output_dir, best_model):
    for checkpoint in os.listdir(output_dir):
        if "checkpoint" in checkpoint and checkpoint != best_model:
            shutil.rmtree(os.path.join(output_dir, checkpoint))


def norm(x):
    return round(x, 3)
