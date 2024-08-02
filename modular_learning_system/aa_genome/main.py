import argparse

import numpy as np
import pyspark
from aa_genome import AA_Genome
from pyspark.sql import SparkSession


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
    args = parser.parse_args()
    return args


def main(args):
    x, y = generate_synthetic_data(1000)

    sc = pyspark.SparkContext()
    spark = SparkSession(sc)

    df = spark.createDataFrame(zip(x.tolist()[::1], y.tolist()), ["x", "y"])

    aa_genome = AA_Genome()
    aa_genome.train_AA_genome_model(df, dimensions=args.dimensions, target_value=args.target_value,
                                    pop_size=args.pop_size, num_generations=args.generations,
                                    mutation_rate=args.mutation_rate)
    best_solution = aa_genome.get_best_solution()

    print(f'Best Solution: {best_solution}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
