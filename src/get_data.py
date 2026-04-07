#get data from the csv to a df for us to run machine learning models
import pandas as pd
def get_data():
    df = pd.read_csv('data/steam_games_dataset.csv')
    return df