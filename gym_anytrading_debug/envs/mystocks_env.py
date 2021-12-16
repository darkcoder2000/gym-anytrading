import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class Actions(Enum):
    """List of possible actions"""
    Sell = 0    
    Hold = 1
    Buy = 2

class MyStocksEnv(gym.Env):
    """Class for trading"""

    def __init__(self, df, window_size, frame_bound, debug):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.debug = debug
        
        # initalize price data
        self.prices, self.signal_features = self.init_data()
        # initalize observation space shape
        self.shape = (window_size, self.signal_features.shape[1])

        # set spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None

        self._last_trade_tick = None  #check if can be removed
        
        self._total_reward = None
        self._total_balance = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._total_balance = 1000.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

        
    def init_data(self):
        #use close values as prices
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size] 
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        #add diff to previous value as seperate column
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def step(self, action):
        self._done = False
        self._current_tick += 1
        
        if self.debug:
            print("step: tick: {0} action: {1} ".format(self._current_tick, Actions(action)))

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True
            print("trade = true")

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        if self.debug:
            print("step: reward: ", step_reward)

        return observation, step_reward, self._done, info