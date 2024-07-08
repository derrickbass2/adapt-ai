from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Function to perform cross-validation
def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return scores