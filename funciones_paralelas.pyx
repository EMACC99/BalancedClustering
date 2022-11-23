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
            


def initialize_distance_matrix(A, centroides, q):
    distance_to_centroids = []
    for elem in A:
        distance_to_centroids.append([np.sqrt((c[0] - elem[0])**2 + (c[1] - elem[1])**2) for c in centroides])

    q.put(distance_to_centroids)    


def get_closest_centroid(distance_to_centroids, q):
    closest_centroid = []
    for elem in distance_to_centroids:
        closest_centroid.append(np.argmin(elem))
    
    q.put(closest_centroid)

def calc_distance_to_centroid(A, centroid, q):
    distance_to_centroid = []
    for elem in A:
        distance_to_centroid.append(np.sqrt((centroid[0] - elem[0]) ** 2 + (centroid[1] - elem[1]) ** 2))

    q.put(distance_to_centroid)

def sum_cluster_weight(capacidades, q):
    q.put(np.sum(capacidades))

