import pyximport

pyximport.install(pyimport=True)

import pandas as pd
import numpy as np
import funciones_paralelas as fp
import queue
import multiprocessing as mp
import sys
import click

from datetime import datetime
from classes import centroide
from typing import Union, List, Tuple
from threading import Thread

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.termination import get_termination


@click.group()
def cli():
    pass


class Clustering_Balandeado(Problem):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        k=4,
        n_var: int,
        n_obj: int,
    ) -> None:

        self.A: np.ndarray = np.column_stack((df["lat"], df["lon"], df["demanda"]))
        self.k = k

        self.distance_matrix: Union[list, np.ndarray] = []

        self.current_centroids: List[centroide] = []  # la sol actual
        self.new_centroids: List[centroide] = []  # la nueva sol
        self.changed_centroid_ix: int = -1  # quien cambio
        self.__changed_centroid_col: Union[list, np.ndarray] = []  # su nueva distancia

        self.__calc_matrix_ranges()

        if k > mp.cpu_count():
            self.__calc_cluster_ranges(k)
            self.eval_std_dist = self.eval_std_dist_less_cpu
        else:
            self.eval_std_dist = self.eval_std_dist_more_cpu

        xl = np.array([df["lat"].min(), df["lon"].min()] * k)
        xu = np.array([df["lat"].max(), df["lon"].max()] * k)

        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def __calc_cluster_ranges(self, k):
        residuo = k % mp.cpu_count()
        paso = k // mp.cpu_count()
        rangos = [_ for _ in range(0, k, paso)]
        for ix, elem in enumerate(rangos):
            rangos[ix] += residuo

        tuplas = [(0, rangos[0])]

        for ix in range(len(rangos) - 1):
            tuplas.append((rangos[ix] + 1, rangos[ix + 1]))

        self.rangos_cluster = tuplas

    def __calc_matrix_ranges(self):
        size = self.A.shape[0]
        residuo = size % mp.cpu_count()
        paso = size // mp.cpu_count()
        rangos = [_ for _ in range(0, size, paso)]
        rangos = rangos[1:]
        for ix, elem in enumerate(rangos):
            rangos[ix] += residuo

        tuplas = [(0, rangos[0])]

        for ix in range(len(rangos) - 1):
            tuplas.append((rangos[ix] + 1, rangos[ix + 1]))

        self.rangos = tuplas

    def eval_std_dist_more_cpu(
        self, centorides: List[centroide]
    ) -> float:  # pasarla a c porque estoy gastando mucho tiempo en wait
        dists = []
        threads: List[Thread] = []
        q: queue.Queue = queue.Queue()

        for c in centorides:
            aux = np.array(c.puntos.copy())

            t = Thread(target=fp.calc_intra_point_distance, args=[aux, q])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        while not q.empty():
            dists.append(q.get())

        return np.std(dists)

    def eval_std_dist_less_cpu(self, centroides: List[centroide]) -> float:
        dists = []
        threads: List[Thread] = []

        q = queue.Queue()
        puntos = [c.puntos for c in centroides]

        for elem in self.rangos_cluster:
            t = Thread(
                target=fp.calc_intra_point_distance_no_cpu,
                args=[puntos[elem[0] : elem[1]], q],
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        while not q.empty():
            dists.append(q.get())

        return np.std(dists)

    def eval_std_weight(
        self, centroides: List[centroide]
    ) -> float:  # esta tambien la tengo que pasar a C
        demandas = []

        for c in centroides:
            demandas.append(np.sum(c.capacidades))

        assert len(demandas) == len(centroides)
        return np.std(demandas)

    def update_distance_matrix(self):  # cambio la sol
        self.distance_matrix[:, self.changed_centroid_ix] = self.__changed_centroid_col
        self.current_centroids = self.new_centroids
        self.new_centroids = []
        self.__changed_centroid_col = []

    def __get_modified_centroid(self, new_centroids: List[centroide]) -> int:
        for ix, elem in enumerate(new_centroids):
            if new_centroids[ix] != self.current_centroids[ix]:
                return ix

    def __one_centroid_changed(self, x: dict):
        self.new_centroids = []
        self.__changed_centroid_col = []
        # x = s["x"]
        for i in range(1, len(x), 2):
            self.new_centroids.append(centroide(x[i - 1], x[i]))

        self.changed_centroid_ix = self.__get_modified_centroid(self.new_centroids)
        changed_centroid = self.new_centroids[self.changed_centroid_ix]

        threads: List[Thread] = []
        q = queue.Queue()
        target = fp.calc_distance_to_centroid

        for _ in self.rangos:
            t = Thread(
                target=target,
                args=[self.A[_[0] : _[1]], [changed_centroid.x, changed_centroid.y], q],
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        while not q.empty():
            self.__changed_centroid_col.extend(q.get())

        aux_distance_matrix = np.zeros_like(self.distance_matrix)

        aux_distance_matrix[:, : self.changed_centroid_ix] = self.distance_matrix[
            :, : self.changed_centroid_ix
        ]
        aux_distance_matrix[:, self.changed_centroid_ix] = self.__changed_centroid_col
        aux_distance_matrix[:, self.changed_centroid_ix :] = self.distance_matrix[
            :, self.changed_centroid_ix :
        ]
        q = queue.Queue()
        target = fp.get_closest_centroid
        threads = []
        closest_centroids = []

        for _ in self.rangos:
            t = Thread(target=target, args=[aux_distance_matrix[_[0] : _[1]], q])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        while not q.empty():
            closest_centroids.extend(q.get())

        for ix, elem in enumerate(closest_centroids):
            a = self.A[ix]
            self.new_centroids[elem].puntos.append((a[0], a[1]))
            self.new_centroids[elem].capacidades.append(a[2])

        return self.eval_std_dist(self.new_centroids), self.eval_std_weight(
            self.new_centroids
        )

    def __calc_all_centroids(
        self, sol: dict, *args, **kwargs
    ) -> Tuple[List[float], List[float]]:
        f1 = []
        f2 = []
        for x in sol:
            for i in range(1, len(x), 2):
                self.current_centroids.append(centroide(x[i - 1], x[i]))

            threads: List[Thread] = []
            q = queue.Queue()

            centroides_coords = [(c.x, c.y) for c in self.current_centroids]
            target = fp.initialize_distance_matrix

            for _ in self.rangos:
                t = Thread(
                    target=target, args=[self.A[_[0] : _[1]], centroides_coords, q]
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            while not q.empty():
                self.distance_matrix.extend(q.get())

            q = queue.Queue()

            target = fp.get_closest_centroid

            threads = []

            closest_centroids = []

            for _ in self.rangos:
                t = Thread(target=target, args=[self.distance_matrix[_[0] : _[1]], q])
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            while not q.empty():
                closest_centroids.extend(q.get())

            for ix, elem in enumerate(closest_centroids):
                a = self.A[ix]
                self.current_centroids[elem].puntos.append((a[0], a[1]))
                self.current_centroids[elem].capacidades.append(a[2])

            self.distance_matrix = np.array(self.distance_matrix)

            f1.append(self.eval_std_dist(self.current_centroids))
            f2.append(self.eval_std_weight(self.current_centroids))
            self.distance_matrix = []

        return f1, f2

    def _evaluate(self, x: dict, out: dict, *args, **kwargs):
        f1, f2 = self.__calc_all_centroids(x)

        out["F"] = np.column_stack([f1, f2])


def read_data(filename: str) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"El conjunto de datos {filename} no fue encontrado")
        sys.exit()

    dataset.sort_values(by=["lat"], inplace=True)
    dataset.drop_duplicates(subset=["lat", "lon"], inplace=True)
    return dataset


@cli.command("run")
@click.option(
    "--pop-size", type=int, required=True, help="The poplulation size for NSGA-II"
)
@click.option("--data", type=str, required=True, help="Path of the dataset to use")
@click.option(
    "--seed",
    type=int,
    required=False,
    default=None,
    help="Seed for the random number generator",
)
@click.option(
    "--n_eval_termination",
    type=int,
    required=False,
    default=300,
    help="Parameter for number of objective function termination",
)
def run(data: str, pop_size: int, seed: int, n_eval_termination: int):
    df = read_data(data)
    problem = Clustering_Balandeado(df, k=4, n_var=8, n_obj=2)

    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_eval", n_eval_termination)
    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)
    dt_string = datetime.now().strftime("%d%m%Y%H%M")

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.save(f"nsga_2_front_{dt_string}")

    with open(f"pareto_nsgaii_{dt_string}.txt", "w+") as f:
        f.write(str(problem.pareto_front()))
        f.write("\n")
        f.write(str(res.F))


cli.add_command(run)

if __name__ == "__main__":
    cli()
