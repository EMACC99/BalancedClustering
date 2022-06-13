import pyximport
pyximport.install(pyimport=True)

import cProfile
import pstats
import pandas as pd
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
    def __init__(self, df : pd.DataFrame, *, k = 4) -> Self:
        
        self.A : np.ndarray = np.column_stack((df["lat"], df["lon"], df['demanda']))

        self.calc_matrix_ranges()
        
        if k > mp.cpu_count():
            self.calc_cluster_ranges(k)
            self.eval_std_dist = self.eval_std_weight_less_cpu
        else:
            self.eval_std_dist = self.eval_std_dist_more_cpu

        super().__init__(num_of_variables=2*k, 
                        types = [AMOSA.Type.REAL] * 2*k, 
                        lower_bounds = [df["lat"].iloc[0], df['lon'].iloc[0]] * k, 
                        upper_bounds = [df["lat"].iloc[-1], df['lon'].iloc[-1]] * k, 
                        num_of_objectives = 2,
                        num_of_constraints = 0)

    def calc_cluster_ranges(self,k):
        residuo = k%mp.cpu_count()
        paso = k//mp.cpu_count()
        rangos = [_ for _ in range(0, k, paso)]
        for ix, elem in enumerate(rangos):
            rangos[ix] += residuo

        tuplas = [(0, rangos[0])]

        for ix in range(len(rangos) -1):
            tuplas.append((rangos[ix] + 1, rangos[ix + 1]))
        
        self.rangos_cluster = tuplas


    def calc_matrix_ranges(self):
        size = self.A.shape[0]
        residuo = size%mp.cpu_count()
        paso = size//mp.cpu_count()
        rangos = [_ for _ in range(0, size, paso)]
        rangos = rangos[1:]    
        for ix, elem in enumerate(rangos):
            rangos[ix] += residuo

        tuplas = [(0, rangos[0])]

        for ix in range(len(rangos) -1):
            tuplas.append((rangos[ix] + 1, rangos[ix + 1]))
        
        self.rangos = tuplas

    def eval_std_dist_more_cpu(self, centorides : List[centroide]) -> float: #pasarla a c porque estoy gastando mucho tiempo en wait
        dists = []
        threads : List[Thread] = []
        q = queue.Queue()

        for c in centorides:
            aux = np.array(c.puntos.copy())
            
            # sums = 0
            # for i in range(1, len(c.puntos)):
            #     sums += fp.distancia(c.puntos[i - 1] , c.puntos[i])
            # dists.append(sums)
            t = Thread(target=fp.calc_intra_point_distance, args=[aux, q])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        
        while not q.empty():
            dists.append(q.get())

        return np.std(dists)

    def eval_std_weight_less_cpu(self, centroides : List[centroide]) -> float:
        dists = []
        threads : List[Thread] = []

        q = queue.Queue()
        puntos = [c.puntos for c in centroides]

        
        for elem in self.rangos_cluster:
            t = Thread(target=fp.calc_intra_point_distance, args=[puntos[elem[0] : elem[1]], q])
        


    def eval_std_weight(self, centroides : List[centroide]) -> float: #esta tambien la tengo que pasar a C
        demandas = []

        # threads = []
        # q = queue.Queue()
        for c in centroides:
            # t = Thread(target=fp.sum_cluster_weight, args=[c.capacidades, q])
            # t.start()
            # threads.append(t)
            demandas.append(np.sum(c.capacidades))

        # while not q.empty():
            # demandas.append(q.get())
        assert (len(demandas) == len(centroides))
        return np.std(demandas)
    
    # def evaluate(self, x : list, out : dict):
    #     centroides : List[centroide] = []
    #     for i in range(1, len(x), 2):
    #         centroides.append(centroide(x[i - 1], x[i]))

    #     for elem in self.__A: #la parte de calcular estas cosas es lo que se esta tardando, tengo que paralelizarla
    #         dist = [distancia(c, (elem[0], elem[1])) for c in centroides] ## O^2, la parte que podria paralelizar es esta
    #         ix = dist.index(min(dist))
    #         centroides[ix].puntos.append((elem[0], elem[1]))
    #         centroides[ix].capacidades.append(elem[2])
        
    #     f1 = self.eval_std_dist(centroides)
    #     f2 = self.eval_std_weight(centroides)
        
    #     out["f"] = [f1, f2]

    def evaluate(self, x : list, out : dict):
        centroides : List[centroide] = []
        for i in range(1, len(x), 2):
            centroides.append(centroide(x[i - 1], x[i]))
        
        threads : List[Thread] = []
        q = queue.Queue()

        centroides_coords = [(c.x,c.y) for c in centroides]
        for elem in self.rangos:
            t = Thread(target=fp.calc_closest_centroid, args=[self.A[elem[0] : elem[1]], centroides_coords ,q])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        
        indices = []
        while not q.empty():
            for elem in q.get():
                indices.append(elem)

        for ix,elem in enumerate(indices):
            a = self.A[ix]
            centroides[elem].puntos.append((a[0], a[1]))
            centroides[elem].capacidades.append(a[2])

        f1 = self.eval_std_dist(centroides)
        f2 = self.eval_std_weight(centroides)

        out["f"] = [f1,f2]



if __name__ == '__main__':
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
        dataset = sys.argv[1] #solo morelos o hidalgo
        dataset = f"INEGI_{dataset}.csv"
    else:
        dataset = "INEGI_morelos.csv"
    try:
        morelos = pd.read_csv(f"data/{dataset}")
    except FileNotFoundError:
        print(f"El conjunto de datos {dataset} no fue encontrado, intente de nuevo")
        sys.exit()
        

    morelos.sort_values(by = ['lat'], inplace = True)

    problem = Clustering_Balandeado(morelos)
    optimizer = AMOSA(config)

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

    hv = get_performance_indicator("hv", ref_point = np.array([max(F[:, 0]) + 10, max(F[:, 1]) + 10]))
    print(f"{hv = }")