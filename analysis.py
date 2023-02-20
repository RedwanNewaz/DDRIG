import os
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum


class PlannerType(Enum):
    SINGLE = 0,
    DDMP = 1

class EnvAnalysis:
    def __init__(self, env_name:str, folder_name:str, num_robots:int):
        self.env_name = env_name
        self.num_robots = num_robots
        self.folder = folder_name
        #

    def dataLoader(self, trajPath, metricPath):
        img_npy = os.path.join('data', 'srtm', 'preprocessed', '%s.npy' % self.env_name)
        self._img = np.load(img_npy)

        traj = pd.read_csv(trajPath)
        xKey = '# x' if '# x' in traj else 'x'
        yKey = ' y'
        self._X = np.array(traj[xKey])
        self._Y = np.array(traj[yKey])
        self._len = self.computePathLen()
        self._metrics = pd.read_csv(metricPath)


    def pathLen(self):
        return self._len

    @classmethod
    def getRobots(cls, env_name:str, folder_name:str, num_robots:int):
        if num_robots > 1:
            result = []
            # data/trajDistributed5/exp-4-robot-N17E073/0_N17E073_distributed_1_ak
            metrics = []
            for filename in sorted(Path(os.path.join('data', folder_name, 'exp-{}-robot-{}'.format(num_robots, env_name))).glob('*/*.csv')):
                metrics.append(filename)

            for i, filename in enumerate (sorted(Path(os.path.join('data', folder_name, 'exp-{}-robot-{}'.format(num_robots, env_name))).glob('*.csv'))):
                newEnv = cls(env_name, folder_name, num_robots)
                newEnv.dataLoader(filename, metrics[i])
                result.append(newEnv)
            return result
        else:
            baseName = 'singleRobotTraj_{}.csv'.format(env_name)
            trajPath = os.path.join('data', folder_name, baseName)

            newEnv = cls(env_name, folder_name, num_robots)
            metricPath = os.path.join('data', folder_name, '0_{}_myopic_ak'.format(env_name), 'metrics.csv' )
            newEnv.dataLoader(trajPath, metricPath)
            return newEnv

    def __getitem__(self, item):
        if item == 'x':
            return self._X
        elif item == 'y':
            return self._Y
        else:
            return self._metrics[item].min()

    def plot(self, item):
        self._metrics[item].plot()

    def computePathLen(self):
        N = len(self._X)
        totalLen = 0
        for i in range(1, N):
            dx = self._X[i] - self._X[i-1]
            dy = self._Y[i] - self._Y[i - 1]
            totalLen += np.sqrt(dx * dx + dy * dy)
        return totalLen


if __name__ == '__main__':
    ENVS = ('N17E073', 'N43W080', 'N45W123', 'N47W124')
    eval_metric = 'rmse'
    srfolder = 'single5'
    num_robots = 6
    ddfolder = 'trajDistributed5%d' % num_robots


    NUM_ENVS = 4 # 'N17E073', 'N43W080', 'N45W123', 'N47W124'
    NUM_COLUMNS = 4 # DDRIG RMSE,  DDRIG Path Len,   RIG RMSE,  RIG Path Len

    data = [[0] * NUM_COLUMNS for _ in range(NUM_ENVS)]
    for i, env_name in enumerate(ENVS):
        singleRIG = EnvAnalysis.getRobots(env_name, srfolder, num_robots=1)
        ddRIG = EnvAnalysis.getRobots(env_name, ddfolder, num_robots=num_robots)

        metrics = [d[eval_metric] for d in ddRIG]
        bestIndex = np.argmin(metrics)

        data[i][0] = metrics[bestIndex]
        data[i][1] = ddRIG[bestIndex].pathLen()
        data[i][2] = singleRIG[eval_metric]
        data[i][3] = singleRIG.pathLen()

        print('[DDRIG] best rmse = ', data[i][0])
        print('[DDRIG] best path len = ', data[i][1])
        print('[RIG] path len = ', data[i][2])
        print('[RIG] eval metric %s = ' % eval_metric, data[i][3])
        print('*' * 100, end='\n\n')

    df = pd.DataFrame(data, columns=['DDRIG RMSE', 'DDRIG Path Len', 'RIG RMSE', 'RIG Path Len'], index=ENVS)
    print(df)

