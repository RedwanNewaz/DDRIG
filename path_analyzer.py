from analysis import EnvAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_ROBTOS = 4
def getDDRIGpaths(env_name):
    ddRIG = EnvAnalysis.getRobots(env_name, ddfolder, num_robots=NUM_ROBTOS)
    metrics = {i:d[eval_metric] for i, d in enumerate(ddRIG)}
    print(metrics)
    bestIndexes = sorted(metrics, key=lambda x : metrics[x])
    print(f'best indexes {bestIndexes}')
    results = []
    FLOOR_SIZE = 3

    for indx in bestIndexes:
        path = np.column_stack((ddRIG[indx]['x'], ddRIG[indx]['y'] ))
        # normalized path
        path = (path) / path.ptp()
        # # floor path
        path = path * FLOOR_SIZE
        results.append(path)
    return results

def minPathDistance(path1, path2):
    minDist = np.inf
    count = 115 + 50
    boundBox = [-2, 2, -2, 2]
    img = EnvAnalysis.getRobots(env_name, ddfolder, num_robots=NUM_ROBTOS)[0]._img
    trajTracker = []
    for j, (A, B) in enumerate(zip(path1, path2)):
        if j < 15:
            continue
        dist = np.linalg.norm(A - B)
        minDist = min(minDist, dist)
        trajTracker.append([A[0], A[1], B[0], B[1]])
        plt.cla()
        plt.imshow(img, extent=boundBox)
        plt.scatter(A[0], A[1])
        plt.scatter(B[0], B[1])
        plt.axis([-2, 2, -2, 2])
        plt.pause(0.1)
        count -= 1
        if count < 1:
            break

    df = pd.DataFrame(trajTracker, columns=['r1x', 'r1y', 'r2x', 'r2y'])
    print(df)
    df.to_csv('results/robotExp/robot-{}-{}.csv'.format(NUM_ROBTOS, env_name))
    return minDist


if __name__ == '__main__':
    ENVS = ('N17E073', 'N43W080', 'N45W123', 'N47W124')
    ddfolder = 'trajDistributed5%d' % NUM_ROBTOS
    eval_metric = 'rmse'
    env_name = ENVS[0]

    paths = getDDRIGpaths(env_name)
    safeDist = minPathDistance(paths[0], paths[1])
    print(safeDist)


