from .models import LRModel, RFModel, SVMModel
from .data_processing import preprocess_text, tfidf_transform, engineer_features
from .training import train_and_evaluate
from .prediction import make_predictions

# Load and preprocess data
data_path = "/Users/derrickbass/Desktop/adaptai/datasets/dataset.csv"  # Replace with your actual data file path
data = pd.read_csv(data_path)

text_data = [preprocess_text(text) for text in data['text_column']]  # Replace 'text_column' with the actual text column name

# Perform TF-IDF vectorization
vectorizer, tfidf_matrix = tfidf_transform(text_data)

# Engineer additional features
X = engineer_features(tfidf_matrix, data[['feature1', 'feature2']])  # Replace with actual feature column names
y = data['target_column']  # Replace 'target_column' with the actual target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
lr_model = LRModel()
rf_model = RFModel()
svm_model = SVMModel()

lr_accuracy = train_and_evaluate(lr_model, X_train, y_train, X_test, y_test)
rf_accuracy = train_and_evaluate(rf_model, X_train, y_train, X_test, y_test)
svm_accuracy = train_and_evaluate(svm_model, X_train, y_train, X_test, y_test)

# Print model accuracies
print(f"LR Model Accuracy: {lr_accuracy}")
print(f"RF Model Accuracy: {rf_accuracy}")
print(f"SVM Model Accuracy: {svm_accuracy}")
