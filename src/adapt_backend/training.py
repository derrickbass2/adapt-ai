from sklearn.model_selection import cross_val_score


# Function to perform cross-validation
def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return scores
