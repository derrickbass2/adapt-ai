import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='ADAPTAI CLI')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output file path')
    return parser.parse_args()
