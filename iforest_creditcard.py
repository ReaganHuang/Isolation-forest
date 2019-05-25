import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sys
import iforest

def auc_score(file):
    df = pd.read_csv(file)
    N = 15000
    df = df.sample(N)
    X, y = df.drop('Class', axis=1), df['Class']

    it = iforest.IsolationTreeEnsemble(sample_size=256, n_trees=300)
    it.fit(X)
    scores = it.anomaly_score(X)

    y_pred = it.predict_from_anomaly_scores(scores, threshold=0.5)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score

file_name = sys.argv[1]
if __name__ == '__main__':
    print(auc_score(file_name))

