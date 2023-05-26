"""
Script for running an experiment end-to-end on a given test dataset batch
  (for all edit algorithms and the unedited model).

What it does:
1) RUN:
    - iterate over all selected models (plus the unedited model)
        - run model on all test samples
        - store output on hard disk
2) SCORE:
    - load output for unedited model from HD
    - iterate over all selected models
        - load output from HD
        - compute scores from edited model output plus unedited model output
        - collect scores
    - store results for all models (incl. unedited model) in csv files on hard disk
"""
import pandas as pd
from typing import Union

from datasets import tqdm

from experiments.evaluate import main as run, get_run_dir
from experiments.e2e_analyze import get_case_df


def score(
        alg_names: list[str],
        model_name: Union[str, tuple],
        start_index: int,
        dataset_size_limit: int,
) -> None:
    def rename_algo(algo):
        if algo == "IDENTITY":
            return model_name
        return algo

    algo_to_run_dir = {
        rename_algo(algo): get_run_dir(
            dir_name=algo,
            model_name=model_name,
            start_index=start_index,
            dataset_size_limit=dataset_size_limit,
        )
        for algo in alg_names
    }
    #full_df = pd.concat(
    #    get_case_df(case_id, algo_to_run_dir=algo_to_run_dir, model_name=model_name)
    #    for case_id in tqdm(range(start_index, start_index + dataset_size_limit))
    #)
    #rewrite the above to ignore case ids which dont exist
    full_df = pd.concat(
        get_case_df(case_id, algo_to_run_dir=algo_to_run_dir, model_name=model_name)
        for case_id in tqdm(range(start_index, start_index + dataset_size_limit))
        if (algo_to_run_dir[model_name] / f"1_edits-case_{case_id}.json").exists()
    )

    # store to disk
    combined_run_dir = get_run_dir(
            dir_name="combined",
            model_name=model_name,
            start_index=start_index,
            dataset_size_limit=dataset_size_limit,
        )
    combined_run_dir.mkdir(exist_ok=True, parents=True)
    full_df.to_csv(combined_run_dir / "results.csv")
    print(f"Results exported to {combined_run_dir / 'results.csv'}.")


def main(
        alg_names: list[str],
        model_name: Union[str, tuple],
        hparams_fnames: list[str],
        ds_name: str,
        dataset_size_limit: int,
        skip_generation_tests: bool,
        generation_test_interval: int,
        conserve_memory: bool,
        num_edits: int = 1,
        use_cache: bool = False,
        verbose: bool = False,
        start_index: int = 0
):
    assert "IDENTITY" in alg_names, "must also run unedited model (include 'IDENTITY')"
    assert len(alg_names) == len(hparams_fnames), "..."

    for alg_name, hparams_fname in zip(alg_names, hparams_fnames):
        run(
            alg_name=alg_name,
            model_name=model_name,
            hparams_fname=hparams_fname,
            ds_name=ds_name,
            dataset_size_limit=dataset_size_limit,
            skip_generation_tests=skip_generation_tests,
            generation_test_interval=generation_test_interval,
            conserve_memory=conserve_memory,
            dir_name=alg_name,
            num_edits=num_edits,
            use_cache=use_cache,
            verbose=verbose,
            start_index=start_index,
        )

    score(
        alg_names=alg_names,
        model_name=model_name,
        dataset_size_limit=dataset_size_limit,
        start_index=start_index,
    )


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_names",
        choices=["MEMIT", "ROME", "FT", "MEND", "IDENTITY"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<model_name>/<run_id>, "
             "where the run_id is of the form 'run_<start_index>_<dataset_size_limit>'. ",
        required=True,
        nargs='+',
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fnames",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
        nargs='+',
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to n records.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for the dataset. Useful for parallelizing runs.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
             "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
             "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Also print detailed information about the running edit algorithm",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False, verbose=False)
    args = parser.parse_args()

    main(
        alg_names=args.alg_names,
        model_name=args.model_name,
        hparams_fnames=args.hparams_fnames,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        verbose=args.verbose,
        start_index=args.start_index
    )
