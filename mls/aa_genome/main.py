import argparse
import numpy as np
import pyspark
from pyspark.sql import SparkSession

from . import AAGenome  # Import the AAGenome class from the module


def generate_synthetic_data(rows: int):
    x = np.random.rand(rows, 10)
    y = np.sin(x[:, 0]) + np.random.normal(scale=0.1, size=rows)
    return x, y


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_value", type=float, default=0.5)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    return parser.parse_args()


def main(parsed_args):
    x, y = generate_synthetic_data(1000)

    sc = pyspark.SparkContext()
    spark = SparkSession(sc)

    df = spark.createDataFrame(zip(x.tolist()[::1], y.tolist()), ["x", "y"])

    # Instantiate AAGenome and train the model
    aa_genome_instance = AAGenome(df, dimensions=parsed_args.dimensions, target_value=parsed_args.target_value,
                                  pop_size=parsed_args.pop_size, num_generations=parsed_args.generations,
                                  mutation_rate=parsed_args.mutation_rate)
    model = aa_genome_instance.train_aa_genome_model()
    best_solution = model.get_best_solution() if model else None

    print(f'Best Solution: {best_solution}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
