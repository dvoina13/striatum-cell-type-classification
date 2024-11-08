import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Template")

    parser.add_argument('-s', '--seed', type=int, default = 2, help="seed")
    
    parser.add_argument('-ns', '--num_selections', type=int, default = None, help="num_selections")
    parser.add_argument('-il', '--input_layer', type=str, default = "binary_gates", help="seed")
    
    parser.add_argument('-bs', '--batch_size', type=int, default = 1, help="batch_size")
    return parser.parse_args()

