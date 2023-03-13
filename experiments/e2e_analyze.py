import os
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

sns.set_context("talk")
sns.set_style("darkgrid")

seedbank.initialize(SEED)

ALIASES = {
    "gpt2-medium": "GPT-2 M",
    "gpt-xl": "GPT-2 XL",
    "gpt2-xl": "GPT-2 XL",
    "EleutherAI/gpt-j-6B": "GPT-J (6B)",
    "gpt-j-6B": "GPT-J (6B)",
    "FT": "FT-L",
}
MODEL_SIZES = {  # according to https://huggingface.co/transformers/v2.2.0/pretrained_models.html
    "gpt2-medium": 345_000_000,
    "gpt2-xl": 1_558_000_000,
    "gpt-j-6b": 6_000_000_000,
}


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


def compute_statistics(df, n_bootstrap: int = 1000) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
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
    mean_ = dfs["mean"]
    mean_.rename(index=ALIASES, inplace=True)
    bootstrap_means_ = dfs["bootstrap_means"]
    bootstrap_means_ = [df.rename(index=ALIASES) for df in bootstrap_means_]
    # list models in inverse order of desired appearance in barplot
    edit_algos = ["MEMIT", "ROME", "FT-L"]
    all_models = edit_algos + [m for m in mean_.index if m not in edit_algos]
    for metric, title, models, suffix in [
        ("S", "Neighborhood Score (NS) ↑", all_models, ""),
        ("M", "Neighborhood Magnitude (NM) ↑", all_models, ""),
        ("KL", "Neighborh. KL divergence (NKL) ↓", edit_algos, ""),
        ("S", "Neighborhood Score (NS) ↑", edit_algos, "simple"),
    ]:
        def post_process_plots(dataset: str = "", metric: str = ""):
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])
            if metric == "KL":
                plt.xscale("log")
                plt.xlim([0, 3 * 1E-5])
            plt.xlabel(title)
            plt.ylabel("")
            plt.tight_layout()
            prefix = f"{metric}_{dataset}" if dataset else metric
            file_name = f"{prefix}_{suffix}.png" if suffix else f"{prefix}.png"
            path = results_dir / file_name
            plt.savefig(path)
            print(f"Exported plot for {metric} to {path}.")

        m, err_ints = prepare_data_for_plots(mean_, bootstrap_means_, metric, models)
        datasets = ["CounterFact", "CounterFact+"]
        col_to_dataset = dict(zip(sorted(m.columns, key=lambda c: "+" in c), datasets))
        m.rename(columns=col_to_dataset, inplace=True)
        err_ints.rename(columns=col_to_dataset, inplace=True)
        m.plot.barh(xerr=err_ints.values)
        post_process_plots(metric=metric)

        # also create barplots for CounterFact and CounterFact+ separately
        for dataset in datasets:
            m[[dataset]].plot.barh(xerr=err_ints[[dataset]].values)
            post_process_plots(metric=metric, dataset=dataset)

        # also create barplots for CounterFact and CounterFact+ in same plot
        m.T.plot.barh()
        post_process_plots(metric=metric, dataset="both")


def prepare_data_for_plots(mean_, bootstrap_means_, metric, models):
    m = pd.DataFrame()
    m[f"{metric}+"] = mean_.loc[models][("N+", metric)]
    m[metric] = mean_.loc[models][("N", metric)]
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
    err_ints = pd.DataFrame(index=["low", "high"])
    for col in m:  # Iterate over bar groups (represented as columns)
        err_ints[col] = [
            m.loc[models, col].values - err_low.loc[models, col].values,
            err_up.loc[models, col].values - m.loc[models, col].values,
        ]
    return m, err_ints


def export_statistics(dfs: Dict[str, pd.DataFrame], results_dir: Path) -> None:
    for key in ["mean", "std", "outliers"]:
        path = results_dir / f"{key}.csv"
        dfs[key].to_csv(path)
        print(f"Exported {key} to {path}.")

    bootstrap_means_dir = results_dir / "bootstrap_means"
    bootstrap_means_dir.mkdir(exist_ok=True)
    for i, df in enumerate(dfs["bootstrap_means"]):
        path = bootstrap_means_dir / f"bootstrap_means_{i:03d}.csv"
        df.to_csv(path)
        print(f"Exported bootstrap sample {i} to {path}.")


def load_statistics(results_dir: Path) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
    bootstrap_means_dir = results_dir / "bootstrap_means"
    bootstrap_files = sorted(bootstrap_means_dir.glob("bootstrap_means_*.csv"))
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


def main(results_dir: Path) -> None:
    """Analyze and plot results for a single model."""
    dfs = get_statistics(results_dir)
    print(format_statistics(dfs))
    plot_statistics(dfs, results_dir)


def get_statistics(results_dir):
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
        dfs = compute_statistics(df)
        export_statistics(dfs, results_dir)
    return dfs


def main_multi(results_dirs: List[Path]) -> None:
    """Analyze and plot results for multiple models."""
    common_parent_dir = Path("/".join(os.path.commonprefix([str(p.resolve()) for p in results_dirs]).split("/")[:-1]))

    model_to_dfs = {}
    means = pd.DataFrame()
    for result_dir in results_dirs:
        dfs = get_statistics(result_dir)
        model = ALIASES[next(alg for alg in dfs["mean"].index if "gpt" in alg)]
        model_to_dfs[model] = dfs

        means_ = dfs["mean"]
        # add model name as an additional index level
        means_.index = pd.MultiIndex.from_product([
            [model],
            [ALIASES.get(alg, alg) if "gpt" not in alg else "Unedited" for alg in means_.index]
        ], names=["model", "algorithm"])
        # sort algorithms by inverse desired order of appearance in the plots
        means_ = means_.reindex(["MEMIT", "ROME", "FT-L", "Unedited"], level="algorithm")
        means = pd.concat([means, means_])

    for metric, title, suffix in [
        ("S", "Neighborhood Score (NS) ↑", ""),
        ("M", "Neighborhood Magnitude (NM) ↑", ""),
        ("KL", "Neighborh. KL divergence (NKL) ↓", ""),
        ("S", "Neighborhood Score (NS) ↑", "simple"),
    ]:
        plt.figure()
        ax = plt.gca()
        for ds in ["N", "N+"]:
            alpha = 0.5 if ds == "N" else 1.
            means[[(ds, metric)]].unstack().plot.bar(rot=0, ax=ax, alpha=alpha)

        handles, labels = ax.get_legend_handles_labels()
        # only use the algorithm names for the legend
        labels = [l.strip("()").split(", ") for l in labels]
        labels = [algo for _, __, algo in labels]
        ax.legend(handles[:3:-1], labels[:3:-1])
        ax.set_ylabel("N" + metric)
        plt.title(title)
        suffix = "_" + suffix if suffix else ""
        path = common_parent_dir / f"means_{metric.lower()}{suffix}.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Exported plot for metric {metric} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        help="Directories to analyze.",
        required=True,
        nargs="+",
    )
    args = parser.parse_args()
    dirs = [Path(d) for d in args.dirs]

    for dir in dirs:
        main_single(results_dir=dir)

    main_multi(dirs)
