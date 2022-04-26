from abc import ABC, abstractmethod


class Recommender(ABC):

    @abstractmethod
    def recommend(self, user):
        pass
