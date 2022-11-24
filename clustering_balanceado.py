import pyximport

pyximport.install(pyimport=True)

import cProfile
import pstats
import pandas as pd
import numpy as np
import funciones_paralelas as fp
import queue
import multiprocessing as mp
import sys

from typing_extensions import Self
from datetime import datetime
from pyAMOSA.AMOSA import *
from typing import Union, List, Tuple
from classes import centroide
from threading import Thread
from pymoo.factory import get_performance_indicator


# def distancia(centroid : Union[centroide, Tuple[float, float]], punto : Tuple[float, float]) -> float :
#     if isinstance(centroid, centroide):
#         return np.sqrt((centroid.x - punto[0])**2 + (centroid.y - punto[1])**2)
#     elif isinstance(centroid, Tuple):
#         return np.sqrt((centroid[0] - punto[0])**2 + (centroid[1] - punto[1])**2)


class Clustering_Balandeado(AMOSA.Problem):
    def __init__(self, df: pd.DataFrame, *, k=4) -> None:

        self.A: np.ndarray = np.column_stack((df["lat"], df["lon"], df["demanda"]))

        self.distance_matrix: Union[list, np.ndarray] = []

        self.current_centroids: List[centroide] = []  # la sol actual
        self.new_centroids: List[centroide] = []  # la nueva sol
        self.changed_centroid_ix: int = -1  # quien cambio
        self.__changed_centroid_col: Union[list, np.ndarray] = []  # su nueva distancia

        self.initialize_archive = True

        self.__calc_matrix_ranges()

        if k > mp.cpu_count():
            self.__calc_cluster_ranges(k)
            self.eval_std_dist = self.eval_std_dist_less_cpu
        else:
            self.eval_std_dist = self.eval_std_dist_more_cpu

        super().__init__(
            num_of_variables=2 * k,
            types=[AMOSA.Type.REAL] * 2 * k,
            lower_bounds=[df["lat"].iloc[0], df["lon"].iloc[0]] * k,
            upper_bounds=[df["lat"].iloc[-1], df["lon"].iloc[-1]] * k,
            num_of_objectives=2,
            num_of_constraints=0,
        )

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
                target=fp.calc_intra_point_distance, args=[puntos[elem[0] : elem[1]], q]
            )

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

    def __get_modified_centroid(self, new_centroids: List[centroide]) -> int:
        for ix, elem in enumerate(new_centroids):
            if new_centroids[ix] != self.current_centroids[ix]:
                return ix

    def __one_centroid_changed(self, s: dict):
        x = s["x"]
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
            self.__changed_centroid_col.append(q.get())

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
            t = Thread(target=target, args=[aux_distance_matrix[_[0], _[1]], q])
            t.start()
            threads.append(t)

        while not q.empty():
            closest_centroids.append(q.get())

        for ix, elem in enumerate(closest_centroids):
            a = self.A[ix]
            self.new_centroids[elem].puntos.append((a[0], a[1]))
            self.new_centroids[elem].capacidades.append(a[2])

        return self.eval_std_dist(self.new_centroids), self.eval_std_weight(
            self.new_centroids
        )

    def __calc_all_centroids(self, s: dict):
        x = s["x"]
        for i in range(1, len(x), 2):
            self.current_centroids.append(centroide(x[i - 1], x[i]))

        threads: List[Thread] = []
        q = queue.Queue()

        centroides_coords = [(c.x, c.y) for c in self.current_centroids]
        target = fp.initialize_distance_matrix

        for _ in self.rangos:
            t = Thread(target=target, args=[self.A[_[0] : _[1]], centroides_coords, q])
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

        return self.eval_std_dist(self.current_centroids), self.eval_std_weight(
            self.current_centroids
        )

    def evaluate(self, s: dict, out: dict):
        if self.initialize_archive:
            f1, f2 = self.__calc_all_centroids(s)
            self.current_centroids = []
            self.distance_matrix = []
        else:
            if isinstance(self.distance_matrix, np.ndarray):
                f1, f2 = self.__one_centroid_changed(s)

            else:
                f1, f2 = self.__calc_all_centroids(s)

        out["f"] = [f1, f2]


if __name__ == "__main__":
    config = AMOSAConfig
    config.archive_hard_limit = 5
    config.archive_soft_limit = 10
    config.archive_gamma = 1
    config.hill_climbing_iterations = 25
    config.initial_temperature = 50
    config.final_temperature = 1
    config.cooling_factor = 0.9
    config.annealing_iterations = 250
    config.early_terminator_window = 15

    if len(sys.argv) == 2:
        dataset = sys.argv[1]  # solo morelos o hidalgo
        dataset = f"INEGI_{dataset}.csv"
    else:
        dataset = "INEGI_morelos.csv"
    try:
        morelos = pd.read_csv(f"data/{dataset}")
    except FileNotFoundError:
        print(f"El conjunto de datos {dataset} no fue encontrado, intente de nuevo")
        sys.exit()

    morelos.sort_values(by=["lat"], inplace=True)

    problem = Clustering_Balandeado(morelos)
    optimizer = AMOSA.from_config(config)

    with cProfile.Profile() as pr:
        optimizer.minimize(problem)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    dt_string = datetime.now().strftime("%d%m%Y%H:%M")
    stats.dump_stats(f"profiler_{dt_string}.prof")
    optimizer.save_results(problem, f"clustering_{dt_string}.csv")
    optimizer.plot_pareto(problem, f"clustering_{dt_string}.pdf")

    F = optimizer.pareto_front()

    hv = get_performance_indicator(
        "hv", ref_point=np.array([max(F[:, 0]) + 10, max(F[:, 1]) + 10])
    )
    print(f"{hv = }")
