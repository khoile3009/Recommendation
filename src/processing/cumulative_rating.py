import pandas as pd
from src.processing.load_data import load_dataframe


def generate_cumulative_rating():
    dataframe = load_dataframe("ratings")

    total_rating = dataframe[['movieId', 'rating']].groupby('movieId').sum()
    total_rating = total_rating.rename(columns={'rating': 'total_rating'})

    num_rating = dataframe[['movieId', 'rating']].groupby('movieId').count()
    num_rating = num_rating.rename(columns={'rating': 'num_rating'})

    result_dataframe = total_rating.join(num_rating, on='movieId')
    result_dataframe['average_rating'] = result_dataframe['total_rating'] / result_dataframe['num_rating']
    result_dataframe.to_csv('dataset/ml-25m/ml-25m/rating_stats.csv')

if __name__ == '__main__':
    generate_cumulative_rating()
