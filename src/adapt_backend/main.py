import os

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from .data_processing import preprocess_text, tfidf_transform, engineer_features
from .database_models import LRModel, RFModel, SVMModel
from .training import train_and_evaluate


def load_and_concatenate_datasets(data_paths):
    """
    Load and concatenate datasets from the provided file paths.

    Args:
        data_paths (list): List of file paths to CSV datasets.

    Returns:
        DataFrame: A concatenated DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If a file path does not exist.
        pd.errors.ParserError: If there is an issue reading any CSV file.
    """
    data_frames = []
    for path in data_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            data_frames.append(pd.read_csv(path))
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Error parsing file: {path}") from e

    return pd.concat(data_frames, ignore_index=True)


def main():
    # List of file paths to all datasets
    data_paths = [
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/data.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/blinkit_retail.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/olist_customers_dataset.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/olist_order_items_dataset.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/olist_orders_dataset.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/olist_sellers_dataset.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/product_category_name_translation.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/retail/Amazon/shopping_trends.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/psych/shopping_behavior_updated.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/psych/garments_worker_productivity.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/hospitality/hotaling_cocktails.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/hospitality/hospitality_employees.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/hospitality/all_drinks.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/hospitality/data_cocktails.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/finance/ed_stats_country.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/finance/ed_stats_country_series.csv",
        "/Users/dbass/PycharmProjects/adapt-ai-real/datasets/finance/ed_stats_series.csv"
    ]

    # Load and combine datasets
    try:
        data = load_and_concatenate_datasets(data_paths)
        print(f"Loaded {len(data)} rows from {len(data_paths)} datasets.")
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error loading datasets: {e}")
        return

    # Handle missing columns
    if 'text_column' not in data.columns or 'target_column' not in data.columns:
        print("Error: Required columns ('text_column', 'target_column') are missing in the data.")
        return

    if 'feature1' not in data.columns or 'feature2' not in data.columns:
        print("Error: Required feature columns ('feature1', 'feature2') are missing in the data.")
        return

    # Preprocess text data
    try:
        # Replace 'text_column' with the actual column name in your datasets
        text_data = [preprocess_text(text) for text in data['text_column']]
        print(f"Processed {len(text_data)} text entries.")
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return

    # Perform TF-IDF vectorization
    try:
        vectorizer, tfidf_matrix = tfidf_transform(text_data)
        print(f"TF-IDF vectorization complete with shape: {tfidf_matrix.shape}.")
    except Exception as e:
        print(f"Error during TF-IDF vectorization: {e}")
        return

    # Engineer additional features
    try:
        # Replace 'feature1' and 'feature2' with actual column names containing numeric features
        X_features = engineer_features(data[['feature1', 'feature2']])
        print("Engineered additional features.")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return

    # Combine features
    try:
        X = pd.concat([pd.DataFrame(tfidf_matrix), X_features], axis=1)
    except Exception as e:
        print(f"Error combining features: {e}")
        return

    # Extract target variable
    try:
        y = data['target_column']  # Replace 'target_column' with the actual target column name
    except KeyError as e:
        print(f"Error: Target column not found in data: {e}")
        return

    # Split data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Split data into training and testing sets. Training size: {len(X_train)}, Testing size: {len(X_test)}.")
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return

    # Initialize models
    lr_model = LRModel()
    rf_model = RFModel()
    svm_model = SVMModel()

    # Train and evaluate models
    try:
        lr_accuracy = train_and_evaluate(lr_model, X_train, y_train, X_test, y_test)
        print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")
    except Exception as e:
        print(f"Error training Logistic Regression Model: {e}")

    try:
        rf_accuracy = train_and_evaluate(rf_model, X_train, y_train, X_test, y_test)
        print(f"Random Forest Model Accuracy: {rf_accuracy:.4f}")
    except Exception as e:
        print(f"Error training Random Forest Model: {e}")

    try:
        svm_accuracy = train_and_evaluate(svm_model, X_train, y_train, X_test, y_test)
        print(f"SVM Model Accuracy: {svm_accuracy:.4f}")
    except Exception as e:
        print(f"Error training SVM Model: {e}")

    # Save trained models
    try:
        lr_model.save('lr_model')
        rf_model.save('rf_model')
        svm_model.save('svm_model')
        print("Models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")


if __name__ == "__main__":
    main()
