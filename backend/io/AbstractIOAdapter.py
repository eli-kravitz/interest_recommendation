from abc import ABC, abstractmethod


class AbstractIO(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def read(self, filename):
        return 0

    @abstractmethod
    def write(self, data):
        return 0
