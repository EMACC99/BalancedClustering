cimport numpy as np
import numpy as np
from typing import Union, List, Tuple
from classes import centroide

def calc_intra_point_distance(puntos, q) -> float:
    suma = 0
    for i in range(1, len(puntos)):
        suma += np.sqrt((puntos[i - 1][0] - puntos[i][0])**2 + (puntos[i - 1][1] - puntos[i][1])**2)
    
    q.put(suma)


def calc_intra_point_distance_no_cpu(lista_puntos, q):
    for puntos in lista_puntos:
        suma= 0
        for i in range(1, len(puntos)):
            suma += np.sqrt((puntos[i - 1][0] - puntos[i][0])**2 + (puntos[i - 1][1] - puntos[i][1])**2)
        q.put(suma)
            


def calc_closest_centroid(A, centroides, q):
    closest_centroid = []
    for elem in A:
        dist = [np.sqrt((c[0] - elem[0])**2 + (c[1] - elem[1])**2) for c in centroides]
        closest_centroid.append(dist.index(min(dist)))
    
    q.put(closest_centroid)

def sum_cluster_weight(capacidades, q):
    q.put(np.sum(capacidades))

