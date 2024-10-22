import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Template")

    parser.add_argument('-s', '--seed', type=int, help="seed")
    return parser.parse_args()

