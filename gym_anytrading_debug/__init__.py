from gym.envs.registration import register
from copy import deepcopy

from . import datasets

register(
    id='stocks-v0',
    entry_point='gym_anytrading_debug.envs:StocksEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)

register(
    id='mystocks-v0',
    entry_point='gym_anytrading_debug.envs:MyStocksEnv',
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)
