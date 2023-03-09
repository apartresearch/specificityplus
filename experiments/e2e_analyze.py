import argparse
import json
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seedbank
import torch
from torch.nn.functional import kl_div
from tqdm import tqdm

from util.globals import *

# TODO: create a proper CLI to replace the following global variables
OUTPUT_DIR = Path(ROOT_DIR / RESULTS_DIR / "plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_context("talk")
sns.set_style("darkgrid")

seedbank.initialize(SEED)


def get_case_df(
        case_id: int,
        algo_to_run_dir: Dict[str, Path],
        model_name: str
) -> pd.DataFrame:
    """
    Return a dataframe summarizing the information about the given test case across all editing algos.

    Extracted information:
    - probability for true token (given test prompt)
    - probability for new token (given test prompt; new token = token from model edit)
    - neg. log probability distribution over all tokens (given test prompt)
    (for all "neighborhood" and "distracting_neighborhood" test prompts)
    """
    # load all information about the given test case into three dictionaries
    #  with `alg` (model edit algo) as the top-level key:
    # - case_results: slightly expanded version of the information used for analysis in the ROME and MEMIT papers
    # - case_probdists: probability distributions over the next token for the different test prompts in this test case
    # - case_metadata: only used for sanity checking
    case_results = {}
    case_metadata = {}
    case_probdists = {}

    case_path = f"1_edits-case_{case_id}.json"
    for alg, run_dir in algo_to_run_dir.items():
        json_dict = json.load((run_dir / case_path).open())
        case_results[alg] = json_dict.pop("post")
        case_metadata[alg] = json_dict

        # load log probs for first token after test prompts
        f_prefix = f"{Path(case_path).stem}_log_probs_"
        case_probdists[alg] = {
            int(prob_file.stem[len(f_prefix):]): torch.load(prob_file)
            for prob_file in sorted(run_dir.glob(f"{f_prefix}*.pt"))
        }

    # sanity check on the metadata; metadata is not used currently, except for this sanity check
    assert all(
        case_metadata[model_name]["requested_rewrite"] == md["requested_rewrite"]
        for md in case_metadata.values()
    )

    # extract info
    df_data_dict = defaultdict(dict)
    for alg, run_dir in algo_to_run_dir.items():
        idx = 0
        # note: to correctly associate the index at the second level in the case_probdists dict
        # with its corresponding test prompt, make sure to iterate over the following keys in the correct order:
        # ["rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs",
        #  "distracting_neighborhood_prompts_probs"]
        for key in ["rewrite", "paraphrase"]:
            pass  # not interested in this info right now;
            idx += len(case_results[alg][f"{key}_prompts_probs"])

        for key, short in {"neighborhood": "N", "distracting_neighborhood": "N+"}.items():
            for x in case_results[alg][f"{key}_prompts_probs"]:
                df_data_dict[(case_id, short, idx)].update({
                    # (alg, "probs_new"): x["target_new"],
                    # (alg, "probs_true"): x["target_true"],
                    (alg, "S"): x["target_true"] < x["target_new"],  # S: success (true object likelier)
                    (alg, "M"): np.exp(-x["target_true"]) - np.exp(-x["target_new"]),  # M: magnitude of pob. diff.
                    (alg, "KL"): kl_div(
                        -case_probdists[model_name][idx], -case_probdists[alg][idx], log_target=True,
                        reduction="batchmean"
                    ).cpu().numpy()
                })
                idx += 1
    df = pd.DataFrame.from_dict(df_data_dict).T
    df.columns.names = ["Algorithm", "Metric"]
    df.index.names = ["Case", "Prompt Type", "Prompt Index"]
    return df


def compute_statistic(df: pd.DataFrame, statistic: Callable) -> Dict[str, pd.DataFrame]:
    # average over all prompts for a given test case and prompt type
    df2 = df.groupby(["Case", "Prompt Type"]).mean().copy()
    # compute statistic across all test cases for a given prompt type
    df2 = df2.groupby("Prompt Type").apply(statistic).stack().T
    # reorder columns
    df2 = df2[[
        ("N", "S"),
        ("N+", "S"),
        ("N", "M"),
        ("N+", "M"),
        ("N", "KL"),
        ("N+", "KL"),
    ]]
    return df2


def get_ranks_for_outlier_detection(df: pd.DataFrame) -> pd.DataFrame:
    # average over all prompts for a given test case and prompt type
    df = df.groupby(["Case", "Prompt Type"]).mean().copy()
    # compute outliers which are detected by KL but not by M or S
    # idea: compute ranks according to KL, M, S
    for (alg, metric) in df.columns:
        df[(alg, f"{metric} rank")] = df[(alg, metric)].rank()
    # ...and compute (rank_M + rank_S)/2 + rank_KL to get test cases which are
    for alg in df.columns.levels[0]:
        #  higher is better for S and M ==> high rank = not suspicious acc. to S and M
        #  lower is better for KL ==> high rank =  suspicious acc. to KL
        # ==> high rank on all of them = suspicious acc. to KL but not acc. to S and M
        df[(alg, "rank Σ")] = (df[(alg, "M rank")] + df[(alg, "M rank")]) / 2 + df[(alg, "KL rank")]
        # high rank: "not suspicious" acc. to M and S but suspicious acc. to KL
    return df


def get_bootstrap_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Draw a hierarchical bootstrap sample (same number of observations=rows as input dataframe) by
    1) resampling (with replacement) the test cases
    2) resampling (with replacement) individual test prompts for each selected test case
    """
    df_by_cases = df.unstack(["Prompt Type", "Prompt Index"]).copy()
    df_by_cases = df_by_cases.sample(len(df_by_cases), replace=True)
    df_by_prompt_type_and_index = df_by_cases.stack(["Prompt Type", "Prompt Index"])
    df_by_prompt_type_and_index = df_by_prompt_type_and_index.sample(len(df_by_prompt_type_and_index), replace=True)
    df_by_prompt_type_and_index = df_by_prompt_type_and_index[df.columns]  # reorder columns to match input
    df_by_prompt_type_and_index = df_by_prompt_type_and_index.astype(df.dtypes.to_dict())  # reset dtypes to match input
    return df_by_prompt_type_and_index


def get_statistics(df, n_bootstrap: int = 1000) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
    dfs = {
        "mean": compute_statistic(df, pd.Series.mean),
        "std": compute_statistic(df, pd.Series.std),
        "outliers": get_ranks_for_outlier_detection(df),
        # compute confidence intervals using bootstrap resampling
        "bootstrap_means": [
            compute_statistic(get_bootstrap_sample(df), pd.Series.mean)
            for _ in tqdm(range(n_bootstrap), desc=f"Drawing {n_bootstrap} bootstrap samples.")
        ],
    }
    return dfs


def format_statistics(dfs: Dict[str, pd.DataFrame]):
    dfs = {
        # TODO: use scientific notation for KL divergence
        key: dfs[key].round(2).astype("str")
        for key in ["mean", "std"]
    }
    return dfs["mean"] + " (" + dfs["std"] + ")"


def plot_statistics(dfs: Dict[str, pd.DataFrame], results_dir: Path):
    model_aliases = {
        "gpt2-medium": "GPT-2 M",
        "gpt-xl": "GPT-2 XL",
        "EleutherAI/gpt-j-6B": "GPT-J (6B)",
        "FT": "FT-L",
    }
    mean_ = dfs["mean"]
    mean_.rename(index=model_aliases, inplace=True)
    bootstrap_means_ = dfs["bootstrap_means"]
    bootstrap_means_ = [df.rename(index=model_aliases) for df in bootstrap_means_]
    all_models = list(mean_.index)
    edit_algos = [m for m in all_models if not m.lower().startswith("gpt")]
    for metric, title, models, suffix in [
        ("S", "Neighborhood Score (NS) ↑", all_models, ""),
        ("M", "Neighborhood Magnitude (NM) ↑", all_models, ""),
        ("KL", "Neighborh. KL divergence (NKL) ↓", edit_algos, ""),
        ("S", "Neighborhood Score (NS) ↑", edit_algos, "simple"),
    ]:
        m, err_ints = prepare_data_for_plots(mean_, bootstrap_means_, metric, models)
        if metric == "KL":
            models = [m for m in models if not m.lower().startswith("gpt")]
            plt.xscale("log")
            plt.xlim([0, 3 * 1E-5])
        m.loc[models].plot.barh(xerr=err_ints)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        relabel = lambda label: "CounterFact+" if "+" in label else "CounterFact"
        ax.legend(handles[::-1], [relabel(l) for l in labels[::-1]])
        plt.xlabel(title)
        plt.ylabel("")
        plt.tight_layout()
        file_name = f"{metric}_{suffix}.png" if suffix else f"{metric}.png"
        path = results_dir / file_name
        plt.savefig(path)
        print(f"Exported plot for {metric} to {path}.")


def prepare_data_for_plots(mean_, bootstrap_means_, metric, models):
    m = pd.DataFrame()
    m[f"{metric}+"] = mean_[("N+", metric)]
    m[metric] = mean_[("N", metric)]
    err_low, err_up = pd.DataFrame(), pd.DataFrame()
    ci = pd.DataFrame([df[("N", metric)] for df in bootstrap_means_]).describe([0.005, 0.995])
    cip = pd.DataFrame([df[("N+", metric)] for df in bootstrap_means_]).describe([0.005, 0.995])
    print(ci)
    print(cip)
    err_low[metric] = ci.loc["0.5%"]
    err_low[f"{metric}+"] = cip.loc["0.5%"]
    err_up[metric] = ci.loc["99.5%"]
    err_up[f"{metric}+"] = cip.loc["99.5%"]
    # prepare confidence intervals for use in pd.DataFrame.plot(xerr=...)
    err_ints = []
    for col in m:  # Iterate over bar groups (represented as columns)
        err_ints.append([
            m.loc[models, col].values - err_low.loc[models, col].values,
            err_up.loc[models, col].values - m.loc[models, col].values,
        ])
    return m, err_ints


def export_statistics(dfs: Dict[str, pd.DataFrame], results_dir: Path) -> None:
    for key in ["mean", "std", "outliers"]:
        path = results_dir / f"{key}.csv"
        dfs[key].to_csv(path)
        print(f"Exported {key} to {path}.")
    for i, df in enumerate(dfs["bootstrap_means"]):
        path = results_dir / f"bootstrap_means_{i:03d}.csv"
        df.to_csv(path)
        print(f"Exported bootstrap sample {i} to {path}.")


def load_statistics(results_dir: Path) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
    bootstrap_files = sorted(results_dir.glob("bootstrap_means_*.csv"))
    if not bootstrap_files:
        raise ValueError("No bootstrap samples found.")

    read_csv = partial(pd.read_csv, header=[0, 1], index_col=[0])
    dfs = {
        "mean": read_csv(results_dir / "mean.csv"),
        "std": read_csv(results_dir / "std.csv"),
        "outliers": read_csv(results_dir / "outliers.csv"),
        "bootstrap_means": [read_csv(path) for path in bootstrap_files],
    }
    return dfs


def concat_results(path) -> pd.DataFrame:
    # loops over all the directories in the path, which each contains a results.csv
    # and concatenates them into one df
    dfs = []
    for p in path.iterdir():
        df = pd.read_csv(p / "results.csv", header=[0, 1], index_col=[0, 1, 2])
        dfs.append(df)
    return pd.concat(dfs)


def main(model_name: str) -> None:
    results_dir = Path(ROOT_DIR / RESULTS_DIR / "combined" / model_name)
    try:
        dfs = load_statistics(results_dir)
    except (ValueError, FileNotFoundError):
        print("Statistics files do not exist, computing statistics...")
        results_file_name = results_dir / "results_combined.csv"
        try:
            df = pd.read_csv(results_file_name, header=[0, 1], index_col=[0, 1, 2])
        except FileNotFoundError:
            print("results_combined.csv does not exist, concatenating results.csv files...")
            df = concat_results(results_dir)
            df.to_csv(results_file_name)
        dfs = get_statistics(df)
        export_statistics(dfs, results_dir)
    print(format_statistics(dfs))
    plot_statistics(dfs, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"],
        default="gpt2-medium",
        help="Model to edit.",
        required=True,
    )
    args = parser.parse_args()

    main(model_name=args.model_name)
