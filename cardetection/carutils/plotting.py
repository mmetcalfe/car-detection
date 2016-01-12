import numpy as np
import matplotlib.pyplot as plt

def saveHistogram(fname, x, bins=50):
    def to_percent(y, pos):
        s = str(100*y)
        if plt.rcParams['text.usetex'] is True:
            return s + r'$\%$'
        else:
            return s + '%'

    # plt.hist(x, bins=bins, normed=True, cumulative=True)
    plt.hist(x, bins=bins)
    # formatter = plt.FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(fname)
    plt.close()
