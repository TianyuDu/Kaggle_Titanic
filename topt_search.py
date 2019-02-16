"""
Feb. 15 2019
Optimize ML Pipeline with Genetic Algorithm
"""
import tpot
from tpot import TPOTClassifier
import data_proc
import sklearn
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = data_proc.parse_data(
        data_proc.load_data("./data/train.csv"))

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    pipeline_optimizer = TPOTClassifier(
        generations=10,
        population_size=20,
        cv=5,
        verbosity=2
    )

    pipeline_optimizer.fit(X_train, y_train)
    # print(pipeline_optimizer.score(X_test, y_test))
    # pipeline_optimizer.export("tpot_test.py")
