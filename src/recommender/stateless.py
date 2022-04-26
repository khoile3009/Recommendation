from src.processing.load_data import load_dataframe
from src.processing.schema import RecommendationResult, User
from src.recommender.base import Recommender


class StatelessRecommender(Recommender):
    def __init__(self):
        self.rating_stats = load_dataframe('rating_stats')

    def recommend(self, user: User) -> RecommendationResult:
        top_rating = self.rating_stats.nlargest(20, 'average_rating')
        return RecommendationResult(top_rating['movieId'].tolist())

if __name__ == '__main__':
    recommender = StatelessRecommender()
    print(recommender.recommend(User(1)))
