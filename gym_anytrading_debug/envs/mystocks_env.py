import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import logging

class Actions(Enum):
    """List of possible actions"""
    Sell = 0    
    Hold = 1
    Buy = 2

# class Wallet:
#     def __init__(self, amount_init_dollars, amount_init_shares):
#         self.dollars = amount_init_dollars
#         self.shares = amount_init_shares


class MyStocksEnv(gym.Env):
    """Class for trading"""

    def __init__(self, df, window_size, frame_bound, debug):
        assert df.ndim == 2

        logging.basicConfig(filename='MyStocksEnv.log', 
            format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', 
            datefmt='%Y-%m-%d,%H:%M:%S', level=logging.DEBUG)

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
        self.start_tick = self.window_size
        self.end_tick = len(self.prices) - 1
        self.done = None
        self.current_tick = None

        self._last_trade_tick = None  #check if can be removed
        
        self.total_reward = None
        
        self.balance = None  
        self.dollars = None
        self.shares = None
        # self._first_rendering = None
        # self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.debug:
            logging.debug("reset")
        self.done = False
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.total_reward = 0.
        self.dollars = 1000.  # unit
        self.shares = 0.
        self.balance = 1000.  
        # self.first_rendering = True
        # self.history = {}
        return self.get_observation()

        
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
        self.current_tick += 1
        
        if self.debug:
            logging.debug("------------------------------------------------")
            logging.debug("step: tick: {0} action: {1} ".format(self.current_tick, Actions(action)))

        if self.current_tick == self.end_tick:
            self.done = True

        step_reward = self.perform_action(action)
        self.total_reward += step_reward

        observation = self.get_observation()
        info = dict(
            total_reward = self.total_reward,
            balance = self.balance,
            dollars = self.dollars,
            shares = self.shares
        )

        if self.debug:
            #logging.debug("reward: {0}".format(step_reward))
            logging.debug(info)

        return observation, step_reward, self.done, info

    def get_balance(self, shares_price):
        return self.dollars + (self.shares * shares_price)

    def perform_action(self, action):
        '''
        Apply the set action by:
         1. Backup current balance
         2. applying action
         3. calculating reward 
            - when action possible then the reward is the balance difference
            - When action is not possible then return 0
        '''
        step_reward = 0

        # 1. Backup current balance
        current_balance = self.balance
        # if self.debug:
        #     logging.debug("current_balance: {0}".format(self.balance))

        # 2. Apply action
        shares_price = self.prices[self.current_tick]
        # if self.debug:
        #     logging.debug("current_shares_price: {0}".format(shares_price))

        if(action == Actions.Buy.value):
            self.buy(shares_price)        
        elif(action == Actions.Hold.value):
            logging.debug("Holding")        
        elif(action == Actions.Sell.value):
            self.sell(shares_price)

        # 3. Calculate reward
        self.balance = self.get_balance(shares_price)
        step_reward =  self.balance - current_balance

        return step_reward

    def buy(self, shares_price):
        if(self.dollars > 0.):
            self.shares = self.dollars / shares_price
            if self.debug:
                logging.debug("Buying at {0} for ${1} getting {2} shares".format(shares_price, self.dollars, self.shares))
            self.dollars = 0.

    def sell(self, shares_price):
        if(self.shares > 0.):
            self.dollars = self.shares * shares_price
            if self.debug:       
                logging.debug("Selling at {0} for {1} shares getting ${2}".format(shares_price, self.shares, self.dollars))
            self.shares = 0.

    def get_observation(self):
        '''returns the observation window'''
        return self.signal_features[(self.current_tick-self.window_size):self.current_tick]

    def render(self, mode='human'):
        logging.debug("render")

    def close (self):
        logging.debug("close")
