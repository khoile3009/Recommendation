from dataclasses import dataclass
from typing import List


@dataclass
class User:
    id: int

@dataclass
class Movie:
    id: int
    name: str

@dataclass
class RecommendationResult:
    movies: List[Movie]
