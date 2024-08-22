import argparse
from mls.spark_engine import SparkEngine


def parse_arguments():
    parser = argparse.ArgumentParser(description='NOMAD CLI')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output file path')
    return parser.parse_args()


def main(parsed_args: argparse.Namespace):
    """
    :param parsed_args: Parsed command-line arguments
    :type parsed_args: argparse.Namespace
    """
    spark_engine = SparkEngine()
    df = spark_engine.read_csv(parsed_args.input)
    spark_engine.write_parquet(df, parsed_args.output)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
