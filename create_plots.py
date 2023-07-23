import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

sns.set_theme()

hue_model_order = ['perceiver-pytorch', 'flash-attention-perceiver']


def calc_relative_improvement(s, col):
    ref_value = s[s['implementation'] == 'perceiver-pytorch'][col].iloc[0]
    rel_improvement = ref_value / s[col]
    return rel_improvement


def create_plot(df, y_col):
    fig, ax = plt.subplots()

    g = sns.barplot(
        df,
        x='input sequence length',
        y=y_col,
        hue='implementation',
        hue_order=hue_model_order,
        width=0.5,
        ax=ax
    )
    g.set_ylim(0, g.get_ylim()[1] * 1.2)
    g.bar_label(g.containers[1])

    return fig

def main(args):
    df = pd.read_csv(args.results_file)
    df = df.rename(columns={
        'model': 'implementation',
        'input_size': 'input sequence length'
    }).sort_values('input sequence length')
    
    for res_col, col in [
        ['speedup', 'time_per_it'],
        ['memory usage reduction', 'peak_memory']
    ]:
        df[res_col] = (
            df
            .groupby(['input sequence length'])
            .apply(calc_relative_improvement, col=col)
            .reset_index(drop=True).values
        )

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)

    out_dir = Path(args.output_dir)

    for y_col in ['speedup', 'memory usage reduction']:
        savename = y_col.replace(' ', '_')

        fig = create_plot(df, y_col)
        fig.savefig(out_dir / f'benchmark_{savename}.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default='benchmark_results.csv')
    parser.add_argument('--output_dir', type=str, default='figures')
    main(parser.parse_args())