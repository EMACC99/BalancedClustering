from typing import List
from typing_extensions import Self


class centroide:
    def __init__(self, x: float, y: float) -> Self:
        self.__x = x
        self.__y = y
        self.puntos = []
        self.capacidades = []
        self.__distancias = []

    def __repr__(self) -> str:
        return f"(x = {self.x}, y = {self.y})"

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def distancias(self) -> List[float]:
        return self.__distancias
