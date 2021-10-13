import numpy as np
np.set_printoptions(precision=2, suppress=True)
np.random.seed(42)

from mcts import MCTS
from stl import STL, PrimitiveGenerator, Simulator

import logging
import sys, os
from datetime import datetime

def thermostat(params) -> Simulator:
    from models.thermostat import Thermostat
    
    tm = Thermostat(out_temp=19, exp_temp=20, latency=2, length=5)
    params['s'] = np.array([[19.53, 19.33, 19.83, 20.08, 19.37]])
    params['range'] = [(0, (19, 21, 20))]
    params['tau'] = 1.0
    return tm

def acas_xu(params) -> Simulator:
    from models.acas_xu import ACAS_XU

    state0 = np.array([5000.0, np.pi/4, -np.pi/2, 300.0, 100.0])
    acasxu = ACAS_XU(state0, tdelta=1.0, slen=10)
    acasxu.load_nnets()
    mins = [0.0, 0.0, -np.pi]
    maxes = [8000.0, np.pi, 0.0]
    params['s'] = acasxu.run()
    params['range'] = [(0, (0, 8000, 16)), (0, (0, np.pi, 8))]
    params['range'] += [(0, (-np.pi, 0, 8)), (1, list(range(5)))]
    params['tau'] = 0.95
    params['epsilon'] = 0.015
    params['max_depth'] = 5
    params['past'] = True
    return acasxu

def auto_transmission(params) -> Simulator:
    from models.auto_transmission import AutoTransmission

    duration = 12
    tdelta = 1.0
    throttles = list(np.linspace(0.6, 0.4, int(duration/tdelta)))
    throttles += [1.0]
    thetas = [0.] * len(throttles)
    at = AutoTransmission(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 5000, 5)), (0, (0, 80, 16))]
    params['range'] += [(0, (0, 1, 10)), (1, list(range(1, 5)))]
    params['epsilon'] = 0.015
    params['tau'] = 0.99
    params['past'] = True
    return at

def auto_transmission2(params) -> Simulator:
    from models.auto_transmission2 import AutoTransmission2

    duration = 12
    tdelta = 0.5
    throttles = list(np.linspace(0.6, 0.4, int(duration/tdelta)))
    throttles += [1.0]
    thetas = [0.] * len(throttles)
    at = AutoTransmission2(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 200, 20)), (0, (0, 1, 10)), (1, [1, 2, 3, 4])]
    params['epsilon'] = 0.015
    params['tau'] = 0.98
    params['past'] = True
    return at

def auto_transmission3(params) -> Simulator:
    from models.auto_transmission3 import AutoTransmission3
    
    tdelta = 1.0
    throttles = [0.55]*5 + [0.9]*5 + [0.55]*6
    thetas = [0.]*len(throttles)

    at = AutoTransmission3(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 6000, 24)), (0, (0, 160, 8))]
    params['epsilon'] = 0.02
    params['tau'] = 1.0
    return at

def auto_transmission4(params) -> Simulator:
    from models.auto_transmission4 import AutoTransmission4
    
    tdelta = 1.0
    throttles = [0.9] * 21
    thetas = [0.]*len(throttles)

    at = AutoTransmission4(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 5000, 5)), (0, (0, 160, 8))]
    params['epsilon'] = 0.02
    params['tau'] = 1.0
    return at

def auto_transmission5(params={}) -> Simulator:
    from models.auto_transmission5 import AutoTransmission5
    
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*6 + [0.1]*6
    thetas = [0.]*len(throttles)

    at = AutoTransmission5(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 4000, 4)), (0, (10, 70, 12))]
    params['tau'] = 1.0
    params['epsilon'] = 0.02
    return at

def auto_transmission6(params={}) -> Simulator:
    from models.auto_transmission6 import AutoTransmission6
    
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*4 + [0.1]*6
    thetas = [0.]*len(throttles)

    at = AutoTransmission6(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 4000, 4)), (0, (10, 70, 12))]
    params['tau'] = 1.0
    params['epsilon'] = 0.02
    return at

def auto_transmission7(params={}) -> Simulator:
    from models.auto_transmission7 import AutoTransmission7
    
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*4 + [0.1]*6
    thetas = [0.]*len(throttles)

    at = AutoTransmission7(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 4000, 4)), (0, (10, 70, 12))]
    params['tau'] = 1.0
    params['epsilon'] = 0.02
    return at


"""
Should be defined in params
---------------------------
s: np.ndarray
    signal being explained
range: list of tuples
    srange[d] = | (0, (min, max, stepsize))     if continuous
                | (1, list of the finite set)   if discrete

Optional
--------
batch_size: int
    number of samples drawn at each rollout
tau: float 
    precision threshold
rho: float
    robustness degree (~coverage) threshold
max_depth: int
    maximum depth to expand the tree
epsilon: float
    maximum tolerated error 
"""

def run(simulator, params={}):
    simulator = eval(simulator)(params)
    if not {'s', 'range'}.issubset(params.keys()):
        logging.error('something undefined in params among {s, range}')
        return
    s           = params.get('s',           None)
    srange      = params.get('range',       None)
    batch_size  = params.get('batch_size',  128 )
    tau         = params.get('tau',         0.95)
    rho         = params.get('rho',         0.01)
    epsilon     = params.get('epsilon',     0.01)
    past        = params.get('past',        False)

    logging.info(f'Signal being analyzed:\n{s}')
    stl = STL()
    primitives = PrimitiveGenerator(s, srange, rho, past).generate()
    logging.info('Initializing primitives...')
    nb = stl.init(primitives, simulator)
    logging.info(f'Done. {nb} primitives.')

    max_depth = params.get('max_depth', 4)
    tree = MCTS(max_depth=max_depth, epsilon=epsilon, tau=tau)
    move = 0
    interrupted = False
    while not interrupted:
        move += 1
        logging.info(f'Move {move}. Choosing best primitive...')
        tree.set_batch_size(move * batch_size)
        try:
            nb = tree.train(stl)
            logging.info(f'{nb} rollouts to reach error {epsilon:5.2%}')
        except KeyboardInterrupt:
            logging.warning('Interrupted')
            interrupted = True
        new_stl = tree.choose(stl)
        logging.info(tree.log(new_stl))
        if tree.anchor_found or len(new_stl) >= max_depth:
            return
        tree.clean(stl, new_stl)
        stl = new_stl


def set_logger(simulator=None):
    formatter = logging.Formatter(
        fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    
    if not simulator:
        os.makedirs('log', exist_ok=True)
        logger = logging.getLogger()
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)
        logger.setLevel(logging.INFO)
        return
    
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    filehandler = logging.FileHandler(f'log/{now}-{simulator}.log', 'a')
    filehandler.setFormatter(formatter)
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    logger.addHandler(filehandler)


def main():
    set_logger()
    simulators = []
    #simulators.append('thermostat')
    #simulators.append('acas_xu')
    simulators.append('auto_transmission')
    #simulators.append('auto_transmission2')
    #simulators.append('auto_transmission3')
    #simulators.append('auto_transmission4')
    #simulators.append('auto_transmission5')
    #simulators.append('auto_transmission6')
    #simulators.append('auto_transmission7')
    for simulator in simulators:
        set_logger(simulator)
        run(simulator)


if __name__ == '__main__':
    main()
