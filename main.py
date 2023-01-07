import sys
import pstats
import cProfile
import click
import pandas as pd
import numpy as np

from datetime import datetime
from clustering_balanceado import Clustering_Balandeado
from pyAMOSA.AMOSA import AMOSA, AMOSAConfig
from pymoo.factory import get_performance_indicator


@click.group()
def cli():
    pass


@cli.command("run")
@click.option("--hard", type=int, required=False, default=75, help="Hard limit")
@click.option("--soft", type=int, required=False, default=150, help="Soft limit")
@click.option("--gamma", type=int, required=False, default=2, help="Gamma")
@click.option(
    "--climb", type=int, required=False, default=2500, help="Hill climbing iterations"
)
@click.option(
    "--itemp", type=float, required=False, default=500, help="Initial temperature"
)
@click.option(
    "--ftemp", type=float, required=False, default=1e-7, help="Final temperature"
)
@click.option("--cool", type=float, required=False, default=0.9, help="Cooling factor")
@click.option(
    "--iter", type=int, required=False, default=2500, help="Annealing iterations"
)
@click.option(
    "--win",
    type=int,
    required=False,
    default=10,
    help="PHY-based early-termination window size",
)
@click.option("-k", type=int, required=True, help="Number of centroids")
@click.option("--data", type=str, required=True, help="path of the dataset to use")
@click.option(
    "--alpha",
    type=float,
    required=False,
    default=1,
    help="Modifier for the hill climber",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    default=None,
    help="seed for the random number generator",
)
def run(
    hard: int,
    soft: int,
    gamma: int,
    climb: int,
    itemp: float,
    ftemp: float,
    cool: float,
    iter: int,
    win: int,
    k: int,
    data: str,
    alpha: float,
    seed: int,
):
    config = AMOSAConfig
    config.archive_hard_limit = hard  # 5
    config.archive_soft_limit = soft  # 10
    config.archive_gamma = gamma  # 1
    config.hill_climbing_iterations = climb  # 25
    config.initial_temperature = itemp  # 50
    config.final_temperature = ftemp  # 1
    config.cooling_factor = cool  # 0.9
    config.annealing_iterations = iter  # 250
    config.early_terminator_window = win  # 15
    config.random_seed = seed

    try:
        dataset = pd.read_csv(data)
    except FileNotFoundError:
        print(f"El conjunto de datos {data} no fue encontrado")
        sys.exit()

    dataset.sort_values(by=["lat"], inplace=True)
    dataset.drop_duplicates(subset=["lat", "lon"], inplace=True)
    problem = Clustering_Balandeado(dataset, k=k, alpha=alpha)
    optimizer = AMOSA.from_config(config)

    with cProfile.Profile() as pr:
        optimizer.minimize(problem)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    dt_string = datetime.now().strftime("%d%m%Y%H%M")
    stats.dump_stats(f"profiler_{dt_string}.prof")
    optimizer.save_results(problem, f"clustering_{dt_string}.csv")
    optimizer.plot_pareto(problem, f"clustering_{dt_string}.pdf")

    F = optimizer.pareto_front()

    hv = get_performance_indicator(
        "hv", ref_point=np.array([max(F[:, 0]) + 10, max(F[:, 1]) + 10])
    )
    print(f"{hv = }")


cli.add_command(run)


if __name__ == "__main__":
    cli()
