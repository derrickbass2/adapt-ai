import argparse
<<<<<<< HEAD

from mls.spark_engine import SparkEngine


=======
from mls.spark_engine import SparkEngine

>>>>>>> cleanup/duplicate-removal
def parse_arguments():
    parser = argparse.ArgumentParser(description='NOMAD CLI')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output file path')
    return parser.parse_args()

<<<<<<< HEAD

=======
>>>>>>> cleanup/duplicate-removal
def main(args):
    spark_engine = SparkEngine()
    df = spark_engine.read_csv(args.input)
    spark_engine.write_parquet(df, args.output)

<<<<<<< HEAD

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
=======
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
>>>>>>> cleanup/duplicate-removal
