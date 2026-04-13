from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_io import read_data
# grab data and put into a panda array (assuming get_data and clean_data have been run already)
df = read_data("steam_games_dataset_clean.db")
# temporarily dropping all string columns until we clean them up
X = df.drop(columns=['estimated_owners'])
X = X.select_dtypes(exclude=['object'])
Y = df['estimated_owners']

# separate data for train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# train model
rc = RandomForestClassifier()
rc.fit(X_train, Y_train)
Y_pred = rc.predict(X_test)
print("done!")