import argparse
import pandas as pd

import experiments.helpers as helpers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', action='store')
    args = parser.parse_args()

    if args.exp == 'mlp':
        df = pd.read_csv('experiments/logs/mlp/for_plot.tsv', sep='\t', header=0)
        helpers.mlp_heatmap(df)

    if args.exp == 'component':
        df = pd.read_csv('experiments/logs/component/for_plot.tsv', sep='\t', header=0)
        helpers.component_heatmap(df)

    if args.exp == 'deprel':
        df = pd.read_csv('experiments/logs/deprel/for_plot.tsv', sep='\t', header=0)
        helpers.deprel_heatmap(df)


main()
