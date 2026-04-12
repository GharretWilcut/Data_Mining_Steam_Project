from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from data_io import read_data
# grab data and put into a panda array
df = read_data("steam_games_dataset.db")
p = df['estimated_owners']
# temporarily dropping all string columns until we clean them up
x = df.drop(columns=['estimated_owners', 'price'])
x = x.select_dtypes(exclude=['object'])
# waiting until estimated owners is cleaned up to include into y
y = df['price']

# separate data for train-test
X_train, X_test, Y_train, Y_test = train_test_split(x, y,)

# train model
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)