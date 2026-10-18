
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

from structure_learning.data import Data
from structure_learning.scores import BGeScore
from structure_learning.evaluation.metrics import JSD, KLD
from structure_learning.distributions.distribution import TrueDistribution
from structure_learning.approximators.missing_data_gibbs import (
    GibbsMissingDataNIWSampler
)
from structure_learning.data import SyntheticDataset, Data
from structure_learning.data_structures import DAG
warnings.simplefilter("ignore", FutureWarning)


# ============================================================
# Configuration
# ============================================================

N_NODES = 3
N_OBSERVATIONS = 500
EDGE_PROBABILITY = 0.3
NOISE_SCALE = 1.0

MISSING_RATES = np.arange(0, 1.0, 0.1)
MISSING_RATES = np.array([0.1,0.3])

N_REPEATS = 3

MCMC_ITERATIONS = 10000
MCMC_BURN_IN = 0.1
GAUSSIAN_ITERATIONS = 20
KNN_NEIGHBORS = 5

RANDOM_SEED = 42

METHODS = [
    "Complete case",
    "Mean",
    "KNN",
    "Gaussian",
    "MCMC"
]

# Set to None for a new experiment.
# Specify an existing run directory to resume instead.
RESUME_DIR = None

BASE_DIR = Path("missing_data_experiment")

if RESUME_DIR is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / timestamp
else:
    output_dir = Path(RESUME_DIR)

output_dir.mkdir(parents=True, exist_ok=True)

KLD_PATH = output_dir / "kld.pkl"
JSD_PATH = output_dir / "jsd.pkl"
MCMC_PATH = output_dir / "mcmc_results.joblib"
DATA_PATH = output_dir / "experiment_data.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(output_dir / "experiment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. Synthetic DAG and data generation
# ============================================================






def generate_synthetic_data(
    n_nodes,
    n_observations,
    degree,
    noise_scale,
    seed
):
    """
    Generate linear-Gaussian data from a random DAG.

    Variables are ordered topologically.

    X_j = sum_i W[i,j] * X_i + epsilon_j
    """
    variables = [chr(ord("a") + i) for i in range(n_nodes)]

    # Simulate a 3-node v-structure (not the equivalnce class)

    incidence = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 0]])
    true_dag = DAG(incidence=incidence, nodes=variables)

    synthetic_data = SyntheticDataset(
        num_nodes=n_nodes,
        num_obs=500,
        node_labels=variables,
        degree=degree,
        noise_std = noise_scale,
        true_dag=true_dag
    )


    # synthetic_data = SyntheticDataset(
    #     num_nodes=n_nodes,
    #     num_obs=500,
    #     node_labels=variables,
    #     degree=degree,
    #     noise_std = noise_scale
    # )

    scaled_data = synthetic_data.data.standardise()

    return synthetic_data.data, scaled_data, variables


# ============================================================
# 2. Generate missingness masks
# ============================================================

def generate_missing_masks(
    data,
    missing_rates,
    n_repeats,
    seed
):
    """
    MCAR missingness:
    Select rows independently with probability missing_rate.
    Remove one randomly chosen variable in each selected row.
    """

    rng = np.random.default_rng(seed)

    n, d = data.shape

    masks = {}
    seeds = np.zeros(
        (len(missing_rates), n_repeats),
        dtype=np.int64
    )

    for i, missing_rate in enumerate(missing_rates):
        for j in range(n_repeats):

            run_seed = int(rng.integers(0, 2**31 - 1))
            seeds[i, j] = run_seed

            local_rng = np.random.default_rng(run_seed)

            mask = np.zeros((n, d), dtype=bool)

            selected_rows = np.flatnonzero(
                local_rng.random(n) < missing_rate
            )

            selected_cols = local_rng.integers(
                0, d, size=len(selected_rows)
            )

            mask[selected_rows, selected_cols] = True

            masks[i, j] = mask

    return masks, seeds


# ============================================================
# 3. Standard imputation methods
# ============================================================

def apply_imputer(data, imputer):
    values = imputer.fit_transform(data)

    return pd.DataFrame(
        values,
        index=data.index,
        columns=data.columns
    )


def complete_case(data, seed=None):
    return data.dropna().copy()


def mean_imputation(data, seed=None):
    return apply_imputer(
        data,
        SimpleImputer(
            strategy="mean",
            keep_empty_features=True
        )
    )


def knn_imputation(data, seed=None):
    return apply_imputer(
        data,
        KNNImputer(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
            keep_empty_features=True
        )
    )


def gaussian_imputation(data, seed=None):
    return apply_imputer(
        data,
        IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=GAUSSIAN_ITERATIONS,
            random_state=seed,
            keep_empty_features=True
        )
    )


def mcmc_imputation(data, variables, seed=None):

    sampler = GibbsMissingDataNIWSampler(
        data=Data(data, variables=variables),
        score_object="bge",
        max_iter=MCMC_ITERATIONS,
        pc_init=False,
        result_type="distribution",
        seed=seed,
        burn_in=MCMC_BURN_IN
    )

    sampler.run()

    return sampler.results


# ============================================================
# 4. Structural posterior and evaluation
# ============================================================

def estimate_distribution(data, variables):

    return TrueDistribution().compute_distribution(
        data,
        score=BGeScore
    )


def evaluate(distribution, ground_truth):
    return {
        "KLD": float(
            KLD().compute(distribution, ground_truth)
        ),
        "JSD": float(
            JSD().compute(distribution, ground_truth)
        )
    }


def run_method(
    method,
    data_missing,
    variables,
    ground_truth,
    seed
):

    if method == "MCMC":
        distribution = mcmc_imputation(
            data_missing,
            variables,
            seed
        )

        return evaluate(distribution, ground_truth), distribution

    imputers = {
        "Complete case": complete_case,
        "Mean": mean_imputation,
        "KNN": knn_imputation,
        "Gaussian": gaussian_imputation
    }

    imputed_data = imputers[method](
        data_missing,
        seed=seed
    )

    if imputed_data.empty:
        raise ValueError("No complete observations available.")

    distribution = estimate_distribution(
        imputed_data,
        variables
    )

    return evaluate(distribution, ground_truth), None


# ============================================================
# 5. Prepare or reload experiment
# ============================================================

if DATA_PATH.exists():

    experiment = joblib.load(DATA_PATH)

    logger.info("Loaded existing experiment data.")

else:

    original_data, scaled_data, variables = (
        generate_synthetic_data(
            n_nodes=N_NODES,
            n_observations=N_OBSERVATIONS,
            degree=2,
            noise_scale=NOISE_SCALE,
            seed=RANDOM_SEED
        )
    )

    missing_masks, seeds = generate_missing_masks(
        original_data,
        MISSING_RATES,
        N_REPEATS,
        seed=RANDOM_SEED + 1
    )



    experiment = {
        "original_data": original_data,
        "scaled_data": scaled_data,
        "variables": variables,
        # "adjacency": adjacency,
        # "weights": weights,
        "missing_rates": MISSING_RATES,
        "n_repeats": N_REPEATS,
        "missing_masks": missing_masks,
        "seeds": seeds,
        "random_seed": RANDOM_SEED
    }

    joblib.dump(
        experiment,
        DATA_PATH,
        compress=3
    )

    logger.info("Generated and saved synthetic experiment.")


original_data = experiment["original_data"].values
scaled_data = experiment["scaled_data"].values
variables = experiment["variables"]
missing_rates = experiment["missing_rates"]
n_repeats = experiment["n_repeats"]
missing_masks = experiment["missing_masks"]
seeds = experiment["seeds"]


# ============================================================
# 6. Ground-truth posterior
# ============================================================

logger.info("Computing ground-truth DAG posterior.")

ground_truth = estimate_distribution(
    original_data,
    variables
)


# ============================================================
# 7. Initialise results and checkpoints
# ============================================================

index = pd.MultiIndex.from_product(
    [missing_rates, range(n_repeats)],
    names=["missing_rate", "iteration"]
)

def load_metrics(path):
    if path.exists():
        df = pd.read_pickle(path).reindex(index)
    else:
        df = pd.DataFrame(index=index, dtype=float)

    for method in METHODS:
        if method not in df.columns:
            df[method] = np.nan

    return df


kld_df = load_metrics(KLD_PATH)
jsd_df = load_metrics(JSD_PATH)

mcmc_results = (
    joblib.load(MCMC_PATH)
    if MCMC_PATH.exists()
    else {}
)


# ============================================================
# 8. Main experiment
# ============================================================

def main():
#TODO: at what stage do we need to scale the data - should we feed raw or scaled data to data imputation methods
    for i, missing_rate in enumerate(missing_rates):
        for j in range(n_repeats):

            key = (i, j)
            idx = (missing_rate, j)

            seed = int(seeds[i, j])

            # Same mask for every method
            data_missing = original_data.mask(
                missing_masks[key]
            )

            for method in METHODS:

                if (
                    pd.notna(kld_df.loc[idx, method])
                    and pd.notna(jsd_df.loc[idx, method])
                ):
                    continue

                start = time.perf_counter()

                logger.info(
                    "Starting rate=%.2f, iteration=%d, method=%s",
                    missing_rate, j, method
                )

                try:

                    if method == "MCMC" and key in mcmc_results:
                        distribution = mcmc_results[key]
                        metrics = evaluate(
                            distribution,
                            ground_truth
                        )

                    else:

                        metrics, distribution = run_method(
                            method,
                            data_missing,
                            variables,
                            ground_truth,
                            seed
                        )

                    if not all(
                        np.isfinite(v) and v >= -1e-10
                        for v in metrics.values()
                    ):
                        raise ValueError(
                            f"Invalid divergence: {metrics}"
                        )

                    # Correct tiny numerical negatives only
                    metrics = {
                        k: max(0.0, v)
                        for k, v in metrics.items()
                    }

                    if method == "MCMC":

                        mcmc_results[key] = distribution

                        joblib.dump(
                            mcmc_results,
                            MCMC_PATH,
                            compress=3
                        )

                    kld_df.loc[idx, method] = metrics["KLD"]
                    jsd_df.loc[idx, method] = metrics["JSD"]

                    # Checkpoint after each method
                    kld_df.to_pickle(KLD_PATH)
                    jsd_df.to_pickle(JSD_PATH)

                    logger.info(
                        "Finished rate=%.2f, iteration=%d, "
                        "method=%s, KLD=%.6f, JSD=%.6f, time=%.1fs",
                        missing_rate,
                        j,
                        method,
                        metrics["KLD"],
                        metrics["JSD"],
                        time.perf_counter() - start
                    )

                except Exception:
                    logger.exception(
                        "Failed rate=%.2f, iteration=%d, method=%s",
                        missing_rate, j, method
                    )

    logger.info("Experiment completed.")
    logger.info("Results saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
