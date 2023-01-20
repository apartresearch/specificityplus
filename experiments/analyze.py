import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seedbank
import torch
from torch.nn.functional import kl_div
from tqdm import tqdm

from util.globals import SEED

# TODO: create a proper CLI to replace the following global variables
MODEL = "gpt2-medium"
UNEDITED_RUN_DIR = Path("results/IDENTITY/run_009")
EDITED_RUN_DIRS = {
    "ROME": Path("results/ROME/run_019"),
    "FT": Path("results/FT/run_003"),
}

CASE_RESULT_FILES = {
    case_id: f"1_edits-case_{case_id}.json"
    for case_id in list(range(1531))
}
OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_context("talk")
sns.set_style("darkgrid")

seedbank.initialize(SEED)


def verify_consistency():
    """Check that the run_dirs contain info about the expected model and test cases."""
    for alg, run_dir in {"IDENTITY": UNEDITED_RUN_DIR, **EDITED_RUN_DIRS}.items():
        assert run_dir.exists()
        for case_id, file in CASE_RESULT_FILES.items():
            json_dict = json.load((run_dir / file).open())
            assert case_id == json_dict["case_id"]
            assert alg == json_dict["alg"]
            assert MODEL == json_dict["model"]


def get_case_df(case_id: int) -> pd.DataFrame:
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
    for alg, run_dir in {MODEL: UNEDITED_RUN_DIR, **EDITED_RUN_DIRS}.items():
        json_dict = json.load((run_dir / CASE_RESULT_FILES[case_id]).open())
        case_results[alg] = json_dict.pop("post")
        case_metadata[alg] = json_dict

        # load log probs for first token after test prompts
        f_prefix = f"{Path(CASE_RESULT_FILES[case_id]).stem}_log_probs_"
        case_probdists[alg] = {
            int(prob_file.stem[len(f_prefix):]): torch.load(prob_file)
            for prob_file in sorted(run_dir.glob(f"{f_prefix}*.pt"))
        }

    # sanity check on the metadata; metadata is not used currently, except for this sanity check
    assert all(
        case_metadata[MODEL]["requested_rewrite"] == md["requested_rewrite"]
        for md in case_metadata.values()
    )

    # extract info
    df_data_dict = defaultdict(dict)
    for alg, run_dir in {MODEL: UNEDITED_RUN_DIR, **EDITED_RUN_DIRS}.items():
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
                        -case_probdists[MODEL][idx], -case_probdists[alg][idx], log_target=True, reduction="batchmean"
                    ).cpu().numpy()
                })
                idx += 1
    df = pd.DataFrame.from_dict(df_data_dict).T
    df.columns.names = ["Algorithm", "Metric"]
    df.index.names = ["Case", "Prompt Type", "Prompt Index"]
    return df


def get_full_results():
    df = pd.concat(get_case_df(case_id) for case_id in CASE_RESULT_FILES)
    return df


def compute_statistic(df: pd.DataFrame, statistic: Callable) -> dict[str, pd.DataFrame]:
    # average over all prompts for a given test case and prompt type
    df = df.groupby(["Case", "Prompt Type"]).mean().copy()
    # compute statistic across all test cases for a given prompt type
    df = df.groupby("Prompt Type").apply(statistic).stack().T
    # reorder columns
    df = df[[
        ("N", "S"),
        ("N+", "S"),
        ("N", "M"),
        ("N+", "M"),
        ("N", "KL"),
        ("N+", "KL"),
    ]]
    return df


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


def get_statistics(df, n_bootstrap: int = 1000) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
    dfs = {
        "mean": compute_statistic(df, pd.Series.mean),
        "std": compute_statistic(df, pd.Series.std),
        "outliers": get_ranks_for_outlier_detection(df),
        # compute confidence intervals using bootstrap resampling
        "bootstrap_means": [
            compute_statistic(df.sample(len(df), replace=True), pd.Series.mean)
            for _ in tqdm(range(n_bootstrap), desc=f"Drawing {n_bootstrap} bootstrap samples.")
        ],
    }
    return dfs


def format_statistics(dfs: dict[str, pd.DataFrame]):
    dfs = {
        # TODO: use scientific notation for KL divergence
        key: dfs[key].round(2).astype("str")
        for key in ["mean", "std"]
    }
    return dfs["mean"] + " (" + dfs["std"] + ")"


def plot_statistics(dfs: dict[str, pd.DataFrame]):
    for metric in ["S", "M", "KL"]:
        m = pd.DataFrame()
        m[metric] = dfs["mean"][("N", metric)]
        m[f"{metric}+"] = dfs["mean"][("N+", metric)]

        err_low, err_up = pd.DataFrame(), pd.DataFrame()
        ci = pd.DataFrame([df[("N", metric)] for df in dfs["bootstrap_means"]]).describe([0.005, 0.995])
        cip = pd.DataFrame([df[("N+", metric)] for df in dfs["bootstrap_means"]]).describe([0.005, 0.995])
        err_low[metric] = ci.loc["0.5%"]
        err_low[f"{metric}+"] = cip.loc["0.5%"]
        err_up[metric] = ci.loc["99.5%"]
        err_up[f"{metric}+"] = cip.loc["99.5%"]

        # prepare confidence intervals for use in pd.DataFrame.plot(xerr=...)
        err_ints = []
        for col in m:  # Iterate over bar groups (represented as columns)
            err_ints.append([m[col].values - err_low[col].values, err_up[col].values - m[col].values])

        m.plot.barh(xerr=err_ints)
        if metric == "KL":
            plt.xscale("log")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{metric}.png")


def export_statistics(dfs: dict[str, pd.DataFrame]):
    for key in ["mean", "std", "outliers"]:
        dfs[key].to_csv(OUTPUT_DIR / f"{key}.csv")


def main():
    RESULTS_FILE_TO_LOAD: Optional[Path] = OUTPUT_DIR / "results.csv"
    if RESULTS_FILE_TO_LOAD and RESULTS_FILE_TO_LOAD.exists():
        df = pd.read_csv(RESULTS_FILE_TO_LOAD, header=[0, 1], index_col=[0, 1, 2])
    else:
        verify_consistency()
        df = get_full_results()
        df.to_csv(OUTPUT_DIR / "results.csv")
    dfs = get_statistics(df)
    print(format_statistics(dfs))
    plot_statistics(dfs)
    export_statistics(dfs)


if __name__ == '__main__':
    main()
