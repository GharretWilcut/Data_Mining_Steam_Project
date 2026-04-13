from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from clean_data import OWNER_COUNT_RANGES
from data_io import read_data

if __name__ == "__main__":
    # grab data and put into a panda array (assuming get_data and clean_data have been run already)
    df_train = read_data("steam_games_dataset_clean_training.db")
    df_test = read_data("steam_games_dataset_clean_testing.db")
    print("Data loaded")
    X_train = df_train.drop(columns=["estimated_owners"])
    Y_train = df_train["estimated_owners"]
    X_test = df_test.drop(columns=["estimated_owners"])
    Y_test = df_test["estimated_owners"]

    # train model
    rc = RandomForestClassifier()
    rc.fit(X_train, Y_train)

    # test model performance
    Y_pred = rc.predict(X_test)

    # provide report
    print(classification_report(Y_test, Y_pred, target_names=OWNER_COUNT_RANGES))

    print("done!")
