import cProfile
import pstats
import pandas as pd
import funciones_paralelas as fp
import queue
import multiprocessing as mp
from pyAMOSA.AMOSA import *
from typing import Union, List, Tuple
from classes import centroide
from threading import Thread


# def distancia(centroid : Union[centroide, Tuple[float, float]], punto : Tuple[float, float]) -> float :
#     if isinstance(centroid, centroide):
#         return np.sqrt((centroid.x - punto[0])**2 + (centroid.y - punto[1])**2)
#     elif isinstance(centroid, Tuple):
#         return np.sqrt((centroid[0] - punto[0])**2 + (centroid[1] - punto[1])**2)


class Clustering_Balandeado(AMOSA.Problem):
    def __init__(self, df : pd.DataFrame, *, k = 4):
        
        self.__A = np.column_stack((df["lat"], df["lon"], df['demanda']))

        super().__init__(num_of_variables=2*k, 
                        types = [AMOSA.Type.REAL] * 2*k, 
                        lower_bounds = [df["lat"].iloc[0], df['lon'].iloc[0]] * k, 
                        upper_bounds = [df["lat"].iloc[-1], df['lon'].iloc[-1]] * k, 
                        num_of_objectives = 2,
                        num_of_constraints = 0)


    def eval_std_dist(self, centorides : List[centroide]) -> float:
        dists = []
        threads = []
        q = queue.Queue()

        for c in centorides: 
            # sums = 0
            # for i in range(1, len(c.puntos)):
            #     sums += fp.distancia(c.puntos[i - 1] , c.puntos[i])
            # dists.append(sums)
            t = Thread(target=fp.calc_intra_point_distance, args=[c.puntos, q])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        
        while not q.empty():
            dists.append(q.get())

        return np.std(dists)
    
    def eval_std_weight(self, centroides : List[centroide]) -> float:
        demandas = []
        for c in centroides:
            demandas.append(np.sum(c.capacidades))
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
        centroides = []
        for i in range(1, len(x), 2):
            centroides.append(centroide(x[i - 1], x[i]))
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

        threads = []
        q = queue.Queue()
        for elem in tuplas:
            t = Thread(target=fp.calc_closest_centroid, args=[self.A[elem[0] : elem[1]], centroides ,q])
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

        f1 = self.eval_std_dist(centroides)
        f2 = self.eval_std_weight(centroides)

        out["f"] = [f1,f2]

    @property
    def A(self) -> np.ndarray:
        return self.__A


if __name__ == '__main__':
    config = AMOSAConfig
    config.archive_hard_limit = 20
    config.archive_soft_limit = 75
    config.archive_gamma = 1
    config.hill_climbing_iterations = 250
    config.initial_temperature = 500
    config.final_temperature = 0.000001
    config.cooling_factor = 0.9
    config.annealing_iterations = 250
    config.early_terminator_window = 15

    morelos = pd.read_csv("data/INEGI_morelos.csv")
    morelos.sort_values(by = ['lat'], inplace = True)

    problem = Clustering_Balandeado(morelos)
    optimizer = AMOSA(config)

    with cProfile.Profile() as pr:
        optimizer.minimize(problem)
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats("profiler.prof")
    optimizer.save_results(problem, "clustering.csv")
    # optimizer.plot_pareto(problem, "clustering.pdf")