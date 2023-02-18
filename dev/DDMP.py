'''
DDMP stands for Distributed Decentralized Motion Planning Algorithm
The key strategy here
    (i) decompose the workspace based on the current state of robots
    (ii) specify task based on the geometric potion area for each robot
    (iii) share model parameters to jointly compute score
'''

from threading import Thread
from time import sleep
class DDMP(Thread):
    def __init__(self, index, models, obs_func, strategy, shared_mutex):
        self.index = index
        self.models = models
        self.make_observation = obs_func
        self.strategy = strategy
        self.shared_mutex = shared_mutex
        self.x_new = None
        self.y_new = None
        self.terminated = False
        Thread.__init__(self)

    def run(self):
        # print(f"[DDMP] Robot {self.index} {self.getName()} is working ...")
        while not self.terminated:

            while not self.strategy.task_assigned:
                sleep(1e-5)
                if self.terminated:
                    break
            try:
                # sample informative locations
                self.x_new = self.strategy.get(model=self.models[self.index])
                # collect observations from informative sample locations
                self.y_new = self.make_observation(self.x_new)
                # add this data point to all models
                # self.shared_mutex.acquire()
                # for m in self.models:
                #     m.add_data(self.x_new, y_new)
                #     m.optimize(num_iter=len(y_new), verbose=False)
                # self.shared_mutex.release()

                # add this information to individual model
                self.shared_mutex.acquire()
                self.models[self.index].add_data(self.x_new, self.y_new)
                # retrain the model
                self.models[self.index].optimize(num_iter=len(self.y_new), verbose=False)
                self.shared_mutex.release()

            except:
                print(f"[DDMP] Exception Robot {self.index}  is terminated prematurelly")
                if self.terminated:
                    break

        print(f'[Robot {self.index}]: thread terminated ...')

