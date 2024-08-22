import argparse

from mls.spark_engine import SparkEngine


def parse_arguments():
    parser = argparse.ArgumentParser(description='NOMAD CLI')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output file path')
    return parser.parse_args()


def main(args):
    spark_engine = SparkEngine()
    df = spark_engine.read_csv(args.input)
    spark_engine.write_parquet(df, args.output)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
