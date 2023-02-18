#!/home/aredwann/anaconda3/envs/rig/bin/python
import os
from pathlib import Path
from time import time
import pypolo
import numpy as np
from threading import Semaphore
from dev.DDMP import DDMP
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from dev.decomposition import voronoi_finite_polygons_2d

from typing import List
from time import sleep


def get_sensor(args, env):
    sensor = pypolo.sensors.Ranger(
        rate=args.sensing_rate,
        env=env,
        env_extent=args.env_extent,
        noise_scale=args.noise_scale,
    )
    return sensor


def get_pilot_data(args, rng, sensor):
    bezier = pypolo.strategies.Bezier(task_extent=args.task_extent, rng=rng)
    x_init = bezier.get(num_states=args.num_init_samples)
    y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)
    return x_init, y_init


def get_robot(x_init, args):
    robot = pypolo.robots.USV(
        init_state=np.array([x_init[0], x_init[1], np.pi / 2]),
        control_rate=args.control_rate,
        max_lin_vel=args.max_lin_vel,
        tolerance=args.tolerance,
        sampling_rate=args.sensing_rate,
    )
    return robot


def get_model(args, x_init, y_init):
    kernel = pypolo.experiments.utilities.get_kernel(args)
    model = pypolo.models.GPR(
        x_train=x_init,
        y_train=y_init,
        kernel=kernel,
        noise=args.init_noise,
        lr_hyper=args.lr_hyper,
        lr_nn=args.lr_nn,
        jitter=args.jitter,
    )
    model.optimize(num_iter=model.num_train, verbose=False)
    return model


def get_evaluator(args, sensor):
    evaluator = pypolo.experiments.Evaluator(
        sensor=sensor,
        task_extent=args.task_extent,
        eval_grid=args.eval_grid,
    )
    return evaluator


def get_strategy(args, rng, robot, index=0):
    """Get sampling strategy."""
    if args.strategy == "random":
        return pypolo.strategies.RandomSampling(
            task_extent=args.task_extent,
            rng=rng,
        )
    elif args.strategy == "active":
        return pypolo.strategies.ActiveSampling(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
        )
    elif args.strategy == "myopic":
        return pypolo.strategies.MyopicPlanning(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
        )
    elif args.strategy == "distributed":
        return pypolo.strategies.DistributedPlanning(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
            index=index,
            alpha = args.alpha
        )
    else:
        raise ValueError(f"Strategy {args.strategy} is not supported.")

def get_voronoi_polygons(robots):
    xy_inits = np.array([robot.state[:2] for robot in robots])
    vor = Voronoi(xy_inits)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = [vertices[region] for region in regions]
    return polygons

def run(args, rng, models, strategies, sensor, evaluators, loggers):

    count = 0
    def animate(trajs:List[np.ndarray], partitions:List[np.ndarray]):
        #increment counter
        nonlocal count
        count += 1

        if not args.no_viz:
            plt.cla()
            plt.imshow(sensor.env.matrix, cmap=plt.cm.gray, interpolation='nearest',
                             extent=sensor.env.extent)
            for x in trajs:
                plt.scatter(x[:, 0], x[:, 1], alpha=0.4)

            for partition in partitions:
                partition = np.vstack((partition, partition[0]))
                plt.plot(partition[:, 0], partition[:, 1])

            plt.axis(args.task_extent)
            plt.pause(1e-2)

        if len(args.save_fig):
            plt.savefig('%s/%04d.png' % (args.save_fig, count))

    obs_func = lambda x_new: sensor.sense(x_new, rng).reshape(-1, 1)
    shared_mutex = Semaphore(len(strategies))

    ddmpThreads = [DDMP(i, models, obs_func, strategy, shared_mutex) for i, strategy in enumerate(strategies)]
    partitions = get_voronoi_polygons([s.robot for s in strategies])

    for i, thread in enumerate(ddmpThreads):
        strategies[i].set(models, partitions[i])
        thread.start()

    while count < args.max_num_samples:
        # check anybody completed its task or not
        assign_new_task = all(not s.task_assigned for s in strategies)

        if assign_new_task:
            # update partition for all robots
            partitions = get_voronoi_polygons([s.robot for s in strategies])
            for i, s in enumerate(strategies):
                s.set(models, partitions[i])
        else:
            trajs = [t.x_new for t in ddmpThreads if t.x_new is not None]
            if not len(trajs):
                continue
            animate(trajs, partitions)
            for i, model in enumerate(models):
                try:
                    print(f'[count = {count}, Model {i}]: ', end="")
                    mean, std, error = evaluators[i].eval_prediction(model)
                    pypolo.experiments.utilities.print_metrics(model, evaluators[i])
                    loggers[i].append(mean, std, error, trajs[i].x_new.copy(), trajs[i].y_new.copy(), model.num_train)
                except:
                    pass
            sleep(1e-5)

    for thread in ddmpThreads:
        thread.terminated = True
    
    for i, thread in enumerate(ddmpThreads):
        print("[-] joining thread ", i + 1)
        thread.join()







def save(args, evaluators, loggers):
    print("Saving metrics and logged data......")
    os.makedirs(args.output_dir, exist_ok=True)
    for i, (evaluator, logger) in enumerate(zip(evaluators, loggers)):
        try:
            experiment_id = "_".join([
                str(args.seed),
                args.env_name,
                args.strategy, str(i + 1),  # i + 1 represents robot index
                args.kernel + args.postfix,
            ])
            save_dir = os.path.join(args.output_dir, experiment_id)
            evaluator.save(save_dir)
            logger.save(save_dir)
        except:
            pass



def get_robots_init_locs(task_extent, N):
    """
        parameters
        ----------
            task_extent: bounding box for target area [xmin, xmax, ymin, ymax]
            N: number of robots
    """
    x_rand = np.random.uniform(task_extent[0], task_extent[1], N)
    y_rand = np.random.uniform(task_extent[2], task_extent[3], N)
    Xinits = np.column_stack([x_rand, y_rand])
    return Xinits

def get_gp_models(args, sensor, rng, Xinits):
    def isValidLocation(x):
        """
        @param x : sample location
        @return true if sample within the bounding box (task_extent)
        """
        xmin, xmax, ymin, ymax = args.task_extent
        return x[0] >= xmin and x[0] < xmax and x[1] >= ymin and x[1] < ymax

    bezier = pypolo.strategies.Bezier(task_extent=args.task_extent, rng=rng)

    models = []
    for x0 in Xinits:
        x_samples = bezier.get(num_states=args.num_init_samples)
        # add random initial location of robot
        x_init = np.array([ x + x0 for x in x_samples if isValidLocation(x + x0)])
        y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)
        # construct gp model with initial samples
        model = get_model(args, x_init, y_init)
        models.append(model)

    return models


def main():
    args = pypolo.experiments.argparser.parse_arguments()
    Xinits = get_robots_init_locs(args.task_extent, args.num_robots)

    rng = pypolo.experiments.utilities.seed_everything(args.seed)
    data_path = "../data/srtm"
    Path(data_path).mkdir(exist_ok=True, parents=True)
    env = pypolo.experiments.environments.get_environment(
        args.env_name, data_path)
    sensor = get_sensor(args, env)

    gpModels = get_gp_models(args, sensor, rng, Xinits)
    robots = [get_robot(x_init, args) for x_init in Xinits]
    strategies = [get_strategy(args, rng, robot, i) for i, robot in enumerate(robots)]

    evaluators = [get_evaluator(args, sensor) for _ in range(args.num_robots)]
    loggers = [pypolo.experiments.Logger(evaluators[i].eval_outputs) for i in range(args.num_robots)]

    start = time()
    run(args, rng, gpModels, strategies, sensor, evaluators, loggers)
    end = time()
    if len(args.output_dir):
        save(args, evaluators, loggers)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()
