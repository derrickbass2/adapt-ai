from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Define your ML models here
class LRModel(LogisticRegression):
    def __init__(self):
        super().__init__(solver='lbfgs', max_iter=1000)

class RFModel(RandomForestClassifier):
    def __init__(self):
        super().__init__(n_estimators=100, random_state=42)

class SVMModel(SVC):
    def __init__(self):
        super().__init__(kernel='rbf', C=1.0, random_state=42)