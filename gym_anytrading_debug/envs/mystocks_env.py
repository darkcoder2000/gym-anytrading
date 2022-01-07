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
    Buy = 1

class MyStocksEnv(gym.Env):
    """Class for trading"""

    def __init__(self, df, window_size, frame_bound, debug, log_level=logging.DEBUG):
        assert df.ndim == 2
        
        self.debug = debug

        logging.basicConfig(filename='MyStocksEnv.log', 
            format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', 
            datefmt='%Y-%m-%d,%H:%M:%S', level=log_level)

        if self.debug:
            logging.info("LogLevel: {0}".format(log_level))
            logging.info(logging.getLevelName(logging.root.level))

        self.seed()
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        
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
        
        self.action_history = None
        
        self.total_reward = None
        self.shares_just_hold = None
        
        self.balance = None  
        self.dollars = None
        self.shares = None
        self.first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.debug:
            logging.info("reset")
        self.done = False
        self.current_tick = self.start_tick
        self.action_history = []
        self.total_reward = 0.
        self.dollars = 1000.  # unit
        self.shares = 0.
        self.balance = 1000.
        self.first_rendering = True
        self.history = {}
        self.shares_just_hold = 1000. / self.prices[self.current_tick]
        if self.debug:
            logging.debug("Initial shares: {0}".format(self.shares_just_hold))
        return self.get_observation()

        
    def init_data(self):
        #use close values as prices
        prices = self.df.loc[:, 'price'].to_numpy()

        prices[self.frame_bound[0] - self.window_size] 
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        #add diff to previous value as seperate column
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def step(self, action):
        self._done = False
        
        if self.debug:
            logging.debug("------------------------------------------------")
            logging.debug("step: tick: {0} action: {1} ".format(self.current_tick, Actions(action)))

        if self.current_tick == self.end_tick:
            self.done = True

        step_reward = self.perform_action(action)
        self.total_reward += step_reward
        
        observation = self.get_observation()

        current_price = self.prices[self.current_tick]

        info = dict(
            total_reward = self.total_reward,
            balance = self.balance,
            dollars = self.dollars,
            shares = self.shares,
            current_price = current_price,
            performance = self.balance / (self.shares_just_hold * current_price)
        )
        self._update_history(info)

        if self.debug:
            logging.debug("reward: {0}".format(step_reward))
            logging.debug(info)
        
        self.current_tick += 1
        return observation, step_reward, self.done, info
       

    def _update_history(self, info):
       if not self.history:
           self.history = {key: [] for key in info.keys()}

       for key, value in info.items():
           self.history[key].append(value)

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
        share_price = self.prices[self.current_tick]
        # if self.debug:
        #     logging.debug("current_shares_price: {0}".format(shares_price))

        if(action == Actions.Buy.value):
            self.buy(share_price)        
        # elif(action == Actions.Hold.value):
        #     logging.info("Holding")        
        elif(action == Actions.Sell.value):
            self.sell(share_price)

        # 3. Calculate reward        
        self.balance = self.get_balance(share_price)
        #performance = self.balance / (self.shares_just_hold * self.prices[self.current_tick])
        step_reward =  self.balance - current_balance

        return step_reward

    def buy(self, shares_price):
        if(self.dollars > 0.):
            self.shares = self.dollars / shares_price
            if self.debug:
                logging.info("Buying at {0} for ${1} getting {2} shares".format(shares_price, self.dollars, self.shares))
            self.dollars = 0.
            self.action_history.append([self.current_tick, Actions.Buy.value])

    def sell(self, shares_price):
        if(self.shares > 0.):
            self.dollars = self.shares * shares_price
            if self.debug:       
                logging.info("Selling at {0} for {1} shares getting ${2}".format(shares_price, self.shares, self.dollars))
            self.shares = 0.
            self.action_history.append([self.current_tick, Actions.Sell.value])

    def get_observation(self):
        '''returns the observation window'''
        return self.signal_features[(self.current_tick-self.window_size):self.current_tick]

    def render(self, mode='human'):
        if self.debug:       
            logging.debug("Render")

    def render_all(self, file_path, mode='human'):
        sell_ticks = []
        buy_ticks = []
        for actionEntry in self.action_history:
            if self.debug:       
                logging.debug("action {0}".format(actionEntry))
            tick = actionEntry[0]
            action = actionEntry[1]
            if  action == Actions.Sell.value:
                sell_ticks.append(tick)
            elif action == Actions.Buy.value:
                buy_ticks.append(tick)

        if self.debug:       
            logging.debug("sell_ticks {0} buy_ticks {1} window_ticks {2}".format(len(sell_ticks),len(buy_ticks), len(self.action_history)))


        # plt.plot(self.prices, 'black')
        # plt.plot(self.history['balance'], 'green')
        # plt.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        # plt.plot(buy_ticks, self.prices[buy_ticks], 'go')

        f = plt.figure(figsize=(25,6))
        ax1 = f.add_subplot()
        color = 'black'
        ax1.set_xlabel('time (days)')
        ax1.set_ylabel('price', color=color)
        ax1.plot(self.prices, color=color)
        ax1.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        ax1.plot(buy_ticks, self.prices[buy_ticks], 'go')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'blue'
        ax2.set_ylabel('balance', color=color)  # we already handled the x-label with ax1

        balance_data = self.history['balance']
        for i in range (0,self.window_size):
            balance_data.insert(0,1000.0)
        ax2.plot(balance_data, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        f.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.balance
        )

        f.savefig(file_path)

        #plt.draw()
        #f.show()


    def close (self):
        logging.debug("close")
