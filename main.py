"""
@file   main.py
@brief  main executable script

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import numpy as np
np.set_printoptions(precision=2, suppress=True)
np.random.seed(42)

from typing import List, Dict, Any

from mcts import MCTS
from stl import STL, Primitive, PrimitiveGenerator
from simulator import Simulator


"""
Create simulators.

Must be defined in params
-------------------------
s: np.ndarray
    signal being explained
range: list of tuples
    srange[d] = (min, max, stepsize)

Optional
--------
tau: float (default = 0.95)
    precision threshold
rho: float (default = 0.03)
    robustness threshold
epsilon: float (default = 0.0075)
    maximum tolerated error 
past: bool (default = False)
    PtSTL or not
batch_size: int (default = 256)
    number of samples drawn at each rollout
max_depth: int (default = 4)
    maximum depth to expand the tree
max_iter: int (default = 50000)
    maximum number of roll-outs
"""

def auto_trans_alarm1(params: Dict[str, Any]) -> Simulator:
    """
    Triggers an alarm when G[0,10](espd<4750) is violated. 
    (Section 5.2)
    """
    from simulators.auto_trans_alarm1 import AutoTransAlarm1
    
    tdelta = 1.0
    throttles = [0.55]*5 + [0.9]*5 + [0.55]*7

    at = AutoTransAlarm1(tdelta, throttles)
    params['s'] = at.run()
    params['range'] = [(0, 6000, 24), (0, 160, 8)]
    params['tau'] = 1.0
    params['epsilon'] = 0.02
    return at

def auto_trans_alarm2(params: Dict[str, Any]) -> Simulator:
    """
    Triggers an alarm when G[0,20](vspd<120) is violated.
    (Section 5.2)
    """
    from simulators.auto_trans_alarm2 import AutoTransAlarm2
    
    tdelta = 1.0
    throttles = [0.9] * 22

    at = AutoTransAlarm2(tdelta, throttles)
    params['s'] = at.run()
    params['range'] = [(0, 5000, 5), (0, 160, 8)]
    params['tau'] = 1.0
    params['epsilon'] = 0.02
    return at

def auto_trans_alarm3(params: Dict[str, Any]) -> Simulator:
    """
    Triggers an alarm when G[0,30](espd<3000) => G[0,4](vspd<35) is violated.
    (Section 5.2)
    """
    from simulators.auto_trans_alarm3 import AutoTransAlarm3
    
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*4 + [0.1]*7

    at = AutoTransAlarm3(tdelta, throttles)
    params['s'] = at.run()
    params['range'] = [(0, 4000, 4), (0, 70, 14)]
    params['tau'] = 1.0
    return at

def auto_trans_alarm4(params: Dict[str, Any]) -> Simulator:
    """
    Triggers an alarm when G[0,30](espd<3000) => G[0,8](vspd<50) is violated.
    (Section 5.2)
    """
    from simulators.auto_trans_alarm4 import AutoTransAlarm4
    
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*4 + [0.1]*7

    at = AutoTransAlarm4(tdelta, throttles)
    params['s'] = at.run()
    params['range'] = [(0, 4000, 4), (0, 70, 14)]
    params['tau'] = 1.0
    return at

def auto_trans_alarm5(params: Dict[str, Any]) -> Simulator:
    """
    Triggers an alarm when G[0,30](espd<3000) => G[0,20](vspd<65) is violated.
    (Section 5.2)
    """
    from simulators.auto_trans_alarm5 import AutoTransAlarm5
    
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*4 + [0.1]*7

    at = AutoTransAlarm5(tdelta, throttles)
    params['s'] = at.run()
    params['range'] = [(0, 4000, 4), (0, 70, 14)]
    params['tau'] = 1.0
    return at

def auto_trans(params: Dict[str, Any]) -> Simulator:
    """
    Simulate an automotive automatic transmission system (Section 5.3).
    This case study aims at explaining the down-shifting (gear 4 to 3) 
    during a passing maneuver.
    """
    from simulators.auto_trans import AutoTrans

    duration = 12
    tdelta = 1.0
    throttles = list(np.linspace(0.6, 0.4, int(duration/tdelta))) + [1.0, 1.0]
    
    at = AutoTrans(tdelta, throttles)
    params['s'] = at.run()
    params['range'] = [(0, 3000, 6), (0, 80, 16), (0, 1, 10)]
    params['tau'] = 0.99
    params['rho'] = 0.01
    params['epsilon'] = 0.0075
    params['past'] = True
    return at

def acas_xu(params: Dict[str, Any]) -> Simulator:
    """
    Simulate an ACAS Xu system (Section 6).
    This case study aims at explaiing the advisory change from 
    Strong Right Turn (SRT) to Weak Right Turn (WRT) for mid-air 
    collision aviodance.
    """
    from simulators.acas_xu import ACAS_XU

    state0 = np.array([5000.0, np.pi/4, -np.pi/2, 300.0, 100.0])
    acasxu = ACAS_XU(state0, tdelta=1.0, slen=10)
    acasxu.load_nnets()
    params['s'] = acasxu.run()
    params['range'] = [(0, 8000, 16), (0, np.pi, 8), (-np.pi, 0, 8)]
    params['tau'] = 0.99
    params['rho'] = 0.01
    params['epsilon'] = 0.0075
    params['max_depth'] = 4
    params['past'] = True
    return acasxu

def thermostat(params: Dict[str, Any]) -> Simulator:
    """
    Simulate an intelligent thermostat (Section 4.3).
    Set-up: outside temperature < expected temperature
            thermostat is off at the beginning
    This case study aims at explaining why the thermostat is off.
    Real explanation: temperature > 20 once within the last two seconds.
    """
    from simulators.thermostat import Thermostat
    
    tm = Thermostat(out_temp=19, exp_temp=20, latency=2, length=5)
    params['s'] = np.array([[19.53, 19.33, 19.83, 20.08, 19.37]])
    params['range'] = [(19, 21, 20)]
    params['tau'] = 1.0
    return tm

"""
Main executable parts.
"""
def run(simulator_name: str) -> None:
    """
    Runs the algorithm for a particular simulator.

    :param simulator_name: function name of the simulator created above,
                           e.g. 'thermostat', 'acas_xu', ...
    """
    params = {}
    simulator = eval(simulator_name)(params)
    if not {'s', 'range'}.issubset(params.keys()):
        logging.error('`s` and `range` should be defined in params')
        return
    s           = params['s']
    srange      = params['range']
    tau         = params.get('tau', 0.95)
    rho         = params.get('rho', 0)
    epsilon     = params.get('epsilon', 0.01)
    past        = params.get('past', False)
    batch_size  = params.get('batch_size', 256)
    max_depth   = params.get('max_depth', 4)
    max_iter    = params.get('max_iter', 50000)

    logging.info(f'Simulator: {simulator_name}')
    logging.info(f'Signal being analyzed:\n{s}')
    logging.info(f'range = {srange}')
    logging.info(f'tau = {tau}')
    logging.info(f'rho = {rho}')
    logging.info(f'epsilon = {epsilon}')
    logging.info(f'batch_size = {batch_size}')
    logging.info(f'max_depth = {max_depth}')
    logging.info(f'max_iter = {max_iter}')
    stl = STL()
    primitives = PrimitiveGenerator(s, srange, rho, past).generate()
    logging.info('Initializing primitives...')
    nb = stl.init(primitives)
    logging.info(f'Done. {nb} primitives.')

    tree = MCTS(simulator, epsilon, tau, batch_size, max_depth, max_iter)
    move = 0
    while True:
        move += 1
        logging.info(f'Move {move}. Choosing best primitive...')
        nb, err = tree.train(stl)
        logging.info(f'{nb} rollouts to reach error {err:5.2%}')
        new_stl = tree.choose(stl)
        if isinstance(new_stl, list):
            logging.info('Maximizing coverage...')
            for anchor in new_stl:
                logging.info(tree.log(anchor))
            return
        logging.info(tree.log(new_stl))
        if tree.finished or len(new_stl) >= max_depth:
            return
        tree.clean(stl, new_stl)
        stl = new_stl

def set_logger(simulator_name: str = None) -> None:
    """
    Log to file in the log folder.

    :param simulator_name: function name of the simulator created above.
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    
    if not simulator_name:
        os.makedirs('log', exist_ok=True)
        logger = logging.getLogger()
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)
        logger.setLevel(logging.INFO)
        return
    
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    filename = os.path.join('log', f'{now}-{simulator_name}.log')
    filehandler = logging.FileHandler(filename, 'a')
    filehandler.setFormatter(formatter)
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    logger.addHandler(filehandler)
    print(f'The following log will be saved to {filename}')

def main(log_to_file: bool = False) -> None:
    "Run algorithm in multiple case studies."
    set_logger() # log to terminal
    simulators = []
    simulators.append('thermostat')
    #simulators.append('auto_trans_alarm1')
    #simulators.append('auto_trans_alarm2')
    #simulators.append('auto_trans_alarm3')
    #simulators.append('auto_trans_alarm4')
    #simulators.append('auto_trans_alarm5')
    #simulators.append('auto_trans')
    #simulators.append('acas_xu')
    for simulator in simulators:
        if log_to_file:
            set_logger(simulator)
        run(simulator)

def empirical_precision(simulator_name: str, 
                        primitives: List[Primitive], 
                        batch_size: int = 10000, 
                        params: Dict[str, Any] = {}):
    """
    A testing function apart from the algorithm. 
    Here we sample multiple times for a particular primitive to estimate its
    empirical precision.
    
    Usage: empirical_precision('acas_xu', [Globally((-4, -1), (0, '<', 4000))])
    """
    simulator = eval(simulator_name)(params)
    STL().init(primitives)
    stl = STL(frozenset(range(len(primitives))))
    Q, N = 0, 0
    for _ in range(batch_size):
        sample, score = simulator.simulate()
        if stl.satisfied(sample):
            Q += score
            N += 1
    print(f'{stl} ({Q}/{N}={Q/N:5.2%})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log to file')
    parser.add_argument('-l', '--log', action='store_true')
    args = parser.parse_args()
    main(args.log)
