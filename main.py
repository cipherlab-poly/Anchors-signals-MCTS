import numpy as np
np.set_printoptions(precision=2, suppress=True)

from src.mcts import MCTS
from src.stl import STL, Generator

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-5s %(message)s',
                    datefmt='%H:%M:%S')

"""
Parameters
----------
s: np.ndarray
    signal being explained
srange: list of tuples
    srange[d] = | (0, (min, max))               if continuous
                | (1, list of the finite set)   if discrete
stepsize: int
    number of evenly spaced signal values for the continuous components
y: int
    black_box(s)
black_box: callable (array -> float)
    black box
tau: float 
    precision threshold
rho: float
    robustness degree (~coverage) threshold
num_train: int
    number of episodes to train MCTS (rollout)
"""
params = {
    's': None,
    'srange': [],
    'y': None,
    'black_box': None,
    'tau': 0.95,
    'rho': 0.01,
}

def explain_thermostat(params):
    '''
    Thermostat
    '''
    from models.thermostat import Thermostat

    slen = 5       # Signal length

    tm = Thermostat(19, 20, 2)
    #tm.simulate(slen)
    tm.temps = [19.53, 19.33, 19.83, 20.08, 19.37]
    tm.on = 0

    params['s'] = np.array([tm.temps])
    params['srange'] = [(0, (19, 21, 20))]
    params['black_box'] = tm.black_box
    params['y'] = tm.on

def explain_acas_xu(params):
    '''
    ACAS-XU
    
    Expect something like [s1 < 5000 ^ -1.5 < s2 < 0.5 ^ s3 < 0 ^ s7 = 3]
    '''
    from models.acasxu.acasxu import ACAS_XU

    slen = 50       # Signal length
    memory = 5      # Length of the latest memory for explanations
    v_own = 100
    v_int = 100
    tau = 0
    a_prev = 0

    inputs0 = np.array([3000.0, np.pi*1.75, -np.pi/2, 100.0, 100.0, 0, 0])
    acas_xu = ACAS_XU(inputs0, slen=105, fault_start=100)
    acas_xu.run()
    smins = acas_xu.nnets[0].mins[:3] + [v_own, v_int, tau, a_prev]
    smaxes = acas_xu.nnets[0].maxes[:3] + [v_own, v_int, tau, a_prev]
    srange = []
    for i in range(3):
        srange.append((0, (smins[i], smaxes[i], 20)))
    srange.append((1, [v_own]))
    srange.append((1, [v_int]))
    srange.append((1, [tau]))
    srange.append((1, range(5)))
    
    params['s'] = acas_xu.signals
    params['srange'] = srange
    params['black_box'] = acas_xu.black_box
    params['y'] = acas_xu.a_actual

def explain_fault_at(params):
    '''
    Automatic transmission fault classification
    '''
    from models.auto_transmission.auto_transmission import AutoTransmission

    duration = 15
    tdelta = 0.5
    slen = int(duration/tdelta)
    throttles = [0.5]*slen
    thetas = [0.]*slen

    at = AutoTransmission(throttles, thetas, tdelta=tdelta)
    at.run(fault2=True)
    
    params['s'] = np.array([at.espds, at.vspds])
    params['srange'] = [(0, (0, 10000, 10)), (0, (0, 300, 10))]
    params['black_box'] = at.black_box
    params['y'] = at.black_box(params['s'])

def get_reward(sample):
    return int(params['black_box'](sample) == params['y'])

def main():
    explain_thermostat(params)
    #explain_acas_xu(params)
    #explain_fault_at(params)
    
    log = ''
    if params['s'] is None:
        log += 'signal s? '
    if not len(params['srange']):
        log += 'range? '
    if params['black_box'] is None:
        log += 'black box? '
    if params['y'] is None:
        params['y'] = black_box(s)
    if len(log):
        logging.error(log)
        return

    logging.info(f"{params['s']} => {params['y']}")
    tree = MCTS(method='fuse')#'uct1tuned')
    stl = STL()
    generator = Generator(params['s'], params['srange'], params['rho'])
    logging.info('Initializing primitives...')
    nb_primitives = stl.initialize(generator, get_reward)
    logging.info(f'Done. {nb_primitives} primitives.')
    while True:
        logging.info('Choosing best primitive...')
        for i in range(2*nb_primitives):
            print(f'{i}/{2*nb_primitives} \r', end='')
            tree.do_rollout(stl)
        stl = tree.choose(stl)
        q, n = tree.qMC[stl], tree.nMC[stl]
        logging.info(f'{stl} ({q}/{n}={q/n:5.2%})')
        if q/n > params['tau']:
            return

if __name__ == '__main__':
    main()
