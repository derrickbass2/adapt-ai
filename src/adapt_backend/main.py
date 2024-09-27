import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from .data_processing import preprocess_text, tfidf_transform, engineer_features
from .models import LRModel, RFModel, SVMModel
from .training import train_and_evaluate

# Load and preprocess data
data_paths = [
    "path/to/amazon_categories.csv",
    "path/to/amazon_data.csv",
    "path/to/blinkit_retail.csv",
    "path/to/olist_customers_dataset.csv",
    "path/to/olist_order_items_dataset.csv",
    "path/to/olist_orders_dataset.csv",
    "path/to/olist_sellers_dataset.csv",
    "path/to/product_category_name_translation.csv",
    "path/to/shopping_trends.csv",
    "path/to/shopping_behavior_updated.csv",
    "path/to/garments_worker_productivity.csv",
    "path/to/hotaling_cocktails.csv",
    "path/to/hospitality_employees.csv",
    "path/to/all_drinks.csv",
    "path/to/data_cocktails.csv",
    "path/to/ed_stats_series.csv",
    "path/to/ed_stats_country_series.csv",
    "path/to/ed_stats_country.csv"
]

data: DataFrame = pd.concat([pd.read_csv(path) for path in data_paths], ignore_index=True)

text_data = [preprocess_text() for text in
             data['text_column']]  # Replace 'text_column' with the actual text column name

# Perform TF-IDF vectorization
vectorizer, tfidf_matrix = tfidf_transform(text_data)

# Engineer additional features
X = engineer_features(data[['feature1', 'feature2']])  # Replace with actual feature column names
y = data['target_column']  # Replace 'target_column' with the actual target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
lr_model = LRModel()
rf_model = RFModel()
svm_model = SVMModel()

# Train and evaluate models
lr_accuracy = train_and_evaluate(lr_model, X_train, y_train, X_test, y_test)
rf_accuracy = train_and_evaluate(rf_model, X_train, y_train, X_test, y_test)
svm_accuracy = train_and_evaluate(svm_model, X_train, y_train, X_test, y_test)

# Print model accuracies
print(f"LR Model Accuracy: {lr_accuracy}")
print(f"RF Model Accuracy: {rf_accuracy}")
print(f"SVM Model Accuracy: {svm_accuracy}")

# Save trained models
lr_model.save('lr_model')
rf_model.save('rf_model')
svm_model.save('svm_model')
