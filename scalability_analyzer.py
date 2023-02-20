from analysis import EnvAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == '__main__':
    ENVS = ('N17E073', 'N43W080', 'N45W123', 'N47W124')

    eval_metric = 'rmse'
    NUM_ENVS = 4 # 'N17E073', 'N43W080', 'N45W123', 'N47W124'
    NUM_COLUMNS = 4 #

    data = [[0] * NUM_COLUMNS for _ in range(NUM_ENVS)]
    for N in range(3, 7):
        for j, env_name in enumerate(ENVS):
            ddfolder = 'trajDistributed5%d' % N
            ddRIG = EnvAnalysis.getRobots(env_name, ddfolder, num_robots=N)
            metrics = [d[eval_metric] for d in ddRIG]
            data[N - 3][j] = min(metrics)


    x = np.arange(4)
    y = np.array(data).T
    # Random test data

    all_data = y
    labels = list(range(3, 7))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))


    # notch shape box plot
    bplot = ax.boxplot(all_data,
                         notch=False,  # notch shape
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    # ax.set_title('Notched box plot')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'cyan']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    plt.errorbar(x + 1, np.mean(y, axis=0), yerr=np.std(y, axis=0), color='k', ls='--')

    ax.yaxis.grid(True)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Mean Absolute Error')
    plt.tight_layout()

    # plt.show()
    plt.savefig('results/errorBars.pgf')