from modular_learning_system.spark_engine import preprocess_data


def load_dataset(s):
    dataset_path = "/Users/derrickbass/Public/adaptai/datasets/hospitality/food101.py"
    return preprocess_data(dataset_path)
