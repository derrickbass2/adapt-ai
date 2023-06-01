from datasets import load_dataset

# Load the food101 dataset
dataset = load_dataset("food101")

# Perform any preprocessing or modifications here

# Save the modified dataset
dataset.save_to_disk("food101_dataset")


