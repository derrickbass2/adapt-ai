# Function to make predictions on new data
def make_predictions(model, trained_model, vectorizer, new_data):
    transformed_data = vectorizer.transform(new_data)
    predictions = trained_model.predict(transformed_data)
    return predictions
