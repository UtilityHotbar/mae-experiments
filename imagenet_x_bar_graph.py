import io
import pandas as pd
import matplotlib.pyplot as plt
from imagenet_x import FACTORS, plots

DATA_PATH = 'logs/imagenet_x_results.txt'

datasets = []

with open(DATA_PATH) as f:
    results = [_.strip() for _ in f.read().split('#')]
    for result in results:
        model_name = result.split('\n')[0]
        data = pd.read_table(io.StringIO('\n'.join(result.split('\n')[1:])), header=0, sep=None)
        datasets.append([model_name, data])

for name, data in datasets:
    plots.plot_bar_plot(data, x='Factor', y='Error ratio')
    plt.show()
