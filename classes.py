from typing import List
from typing_extensions import Self


class centroide:
    def __init__(self, x: float, y: float) -> None:
        self.__x = x
        self.__y = y
        self.capacidades: List[int] = []
        self.puntos: List[tuple] = []

    def __repr__(self) -> str:
        return f"(x = {self.x}, y = {self.y})"

    def __eq__(self, other: Self) -> bool:
        return self.__x == other.__x and self.__y == other.__y

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(self, other)

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y
