import sklearn.ensemble
from sklearn.svm import SVC

# Define your ML models here
class LogisticRegression:
    def __init__(self, max_iter=100):
        self.model = sklearn.linear_model.LogisticRegression(max_iter=max_iter)


class LRModel(LogisticRegression):
    def __init__(self):
        super().__init__(max_iter=1000)


class RFModel(sklearn.ensemble.RandomForestClassifier):
    def __init__(self):
        super().__init__(random_state=42)


class SVMModel(SVC):
    def __init__(self):
        super().__init__(random_state=42)
