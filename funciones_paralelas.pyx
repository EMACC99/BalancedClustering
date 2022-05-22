import numpy as np

from typing import Union, List, Tuple
from classes import centroide

def calc_intra_point_distance(puntos : List[Tuple], q) -> float:
    suma = 0
    for i in range(1, len(puntos)):
        suma += np.sqrt((puntos[i - 1][0] - puntos[i][0])**2 + (puntos[i - 1][1] - puntos[i][1])**2)
    
    q.put(suma)

def calc_closest_centroid(A : np.npdarray, centroides : List[centroide], q) -> List[int]:
    closest_centroid = []
    for elem in A:
        dist = [np.sqrt((c.x - elem[0])**2 + (c.y - elem[1])**2) for c in centroides]
        closest_centroid.append(dist.index(min(dist)))
    
    q.put(closest_centroid)