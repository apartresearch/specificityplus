import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import kl_div

MODEL = "gpt2-medium"
UNEDITED_RUN_DIR = Path("results/IDENTITY/run_008")
EDITED_RUN_DIRS = {
    "ROME": Path("results/ROME/run_018"),
}
CASE_RESULT_FILES = {
    case_id: f"1_edits-case_{case_id}.json"
    for case_id in list(range(10))
}


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


def get_statistics(df) -> dict[str, pd.DataFrame]:
    # average over all prompts for a given test case and prompt type
    df2 = df.groupby(["Case", "Prompt Type"]).mean()
    # compute mean and std over all cases for a given prompt type
    dfs = {}
    for key, statistic in [
        ("mean", pd.DataFrame.mean),
        ("std", pd.DataFrame.std),
    ]:
        dfs[key] = df2.groupby("Prompt Type").apply(statistic).stack().T
        # reorder columns
        dfs[key] = dfs[key][[
            ("N", "S"),
            ("N+", "S"),
            ("N", "M"),
            ("N+", "M"),
            ("N", "KL"),
            ("N+", "KL"),
        ]]
    return dfs


def format_statistics(dfs: dict[str, pd.DataFrame]):
    dfs = {
        key: df.round(2).astype("str")
        for key, df in dfs.items()
    }
    return dfs["mean"] + " (" + dfs["std"] + ")"


def main():
    verify_consistency()
    df = get_full_results()
    dfs = get_statistics(df)
    print(format_statistics(dfs))


if __name__ == '__main__':
    main()
