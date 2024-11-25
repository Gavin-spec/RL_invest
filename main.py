# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:02:57 2024

@author: User
"""

from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import os
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#######################################################################ˇ

#設定環境
class Actions(Enum):
    Buy_NTD = 0
    Buy_S1 = 1
    Buy_S2 = 2
    Buy_S3 = 3
    Buy_S4 = 4
    Buy_S5 = 5
    Buy_S6 = 6
    Buy_S7 = 7
    Buy_S8 = 8

class Positions(Enum):
    # 代表持有幣別
    NTD = 0
    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4
    S5 = 5
    S6 = 6
    S7 = 7
    S8 = 8

    def opposite(self,action):
      return Positions(action)



class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df_list, window_size, render_mode=None):
        # assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.df_list = df_list

        self.render_mode = render_mode
        self.window_size = window_size
        self.prices_list, self.signal_features_list = self._process_data()
        self.shape = (window_size, self.signal_features_list[0].shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices_list[0]) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None

        self._last_position = None
        self._action = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.NTD
        self._position_history = (self.window_size * [None]) + [self._position]
        self._action = 0
        self._total_reward = 0.
        self._total_profit = 10000.  # unit
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        self._action = action
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False

        if action != self._position.value:
            trade = True

        if trade:
            self._last_position = self._position
            self._position = self._position.opposite(action)
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        observation = []
        for i in range(len(self.signal_features_list)):
          observation.extend(self.signal_features_list[i][self._current_tick-self.window_size:self._current_tick])
        return np.array(observation, dtype=np.float32)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def choice_price_col(self, position, buy_or_sell="buy"):
        col_name = 'open' if buy_or_sell == "buy" else 'close'

        foreign_price = None
        if position == Positions.S1:
          foreign_price = self.prices_list[0][col_name].to_numpy()
        elif position == Positions.S2:
          foreign_price = self.prices_list[1][col_name].to_numpy()
        elif position == Positions.S3:
          foreign_price = self.prices_list[2][col_name].to_numpy()
        elif position == Positions.S4:
          foreign_price = self.prices_list[3][col_name].to_numpy()
        elif position == Positions.S5:
          foreign_price = self.prices_list[4][col_name].to_numpy()
        elif position == Positions.S6:
          foreign_price = self.prices_list[5][col_name].to_numpy()
        elif position == Positions.S7:
          foreign_price = self.prices_list[6][col_name].to_numpy()
        elif position == Positions.S8:
          foreign_price = self.prices_list[7][col_name].to_numpy()
        return foreign_price


    def render(self, mode='human'):

        def _plot_position():
            # 有買賣
            if self._action != self._position.value:

              # 現在不是持有台幣(即有買入股票)
              if self._position != Positions.NTD:
                # 買入用紅色
                buy_price_col = self.choice_price_col(self._position)
                plt.scatter(self._current_tick, buy_price_col[self._current_tick], color='red')

              # 上一步不是持有台幣(即有賣出股票)
              if self._last_position != Positions.NTD:
                # 賣出用綠色
                sell_price_col = self.choice_price_col(self._last_position)
                plt.scatter(self._current_tick, sell_price_col[self._current_tick], color='green')
        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices_list[0]['open'].to_numpy(), label="S1")
            plt.plot(self.prices_list[1]['open'].to_numpy(), label="S2")
            plt.plot(self.prices_list[2]['open'].to_numpy(), label="S3")
            plt.plot(self.prices_list[3]['open'].to_numpy(), label="S4")
            plt.plot(self.prices_list[4]['open'].to_numpy(), label="S5")
            plt.plot(self.prices_list[5]['open'].to_numpy(), label="S6")
            plt.plot(self.prices_list[6]['open'].to_numpy(), label="S7")
            plt.plot(self.prices_list[7]['open'].to_numpy(), label="S8")
            # plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.0, 1.0))

            # 起始點標藍色
            plt.scatter(self._current_tick, self.prices_list[0]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[1]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[2]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[3]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[4]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[5]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[6]['open'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices_list[7]['open'].to_numpy()[self._current_tick], color='blue')

        _plot_position()

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Fund: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)


    def render_all(self, title=None):

        plt.cla()
        plt.plot(self.prices_list[0]['open'].to_numpy(), label="S1")
        plt.plot(self.prices_list[1]['open'].to_numpy(), label="S2")
        plt.plot(self.prices_list[2]['open'].to_numpy(), label="S3")
        plt.plot(self.prices_list[3]['open'].to_numpy(), label="S4")
        plt.plot(self.prices_list[4]['open'].to_numpy(), label="S5")
        plt.plot(self.prices_list[5]['open'].to_numpy(), label="S6")
        plt.plot(self.prices_list[6]['open'].to_numpy(), label="S7")
        plt.plot(self.prices_list[7]['open'].to_numpy(), label="S8")
        plt.legend(bbox_to_anchor=(1.0, 1.0))

        last_positions = Positions.NTD

        for i, position in enumerate(self._position_history):
          if position != None:
            # 有買賣
            if position != last_positions:
              # 現在不是持有台幣(即有買入股票)
              if position != Positions.NTD:
                price_col = self.choice_price_col(position)
                plt.scatter(i, price_col[i], color='red')

              # 上一步不是持有台幣(即有賣出股票)
              if last_positions != Positions.NTD:
                price_col = self.choice_price_col(last_positions)
                plt.scatter(i, price_col[i], color='green')

              last_positions = self._position_history[i]

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Fund: %.6f" % self._total_profit
        )

    def render_hold(self, title=None):
        plt.cla()

        # 使用天數作為橫軸
        days = range(len(self._position_history))
        y = self._position_history
        y = [yi.value if yi is not None else 0 for yi in y]
        # y = y.append(y[-1])
        plt.step(days, y)
        plt.xlabel('Day')
        plt.ylabel('Position')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Fund: %.6f" % self._total_profit
        )

        # 顯示圖形
        plt.show()


    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError


class StockEnv(TradingEnv):

    def __init__(self, df_list, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.handing_charge = 0.001425
        self.transaction_tex = 0.0030
        super().__init__(df_list, window_size, render_mode)

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):

        # 有交易
        if action != self._position.value:
          # 原本非台幣
          if self._position != Positions.NTD:
            # 此處賣出為銀行方，等於投資者的買入
            buy_price_col = self.choice_price_col(self._position, "sell")
            buy_price = buy_price_col[self._last_trade_tick]

            # 此處買入為銀行方，等於投資者的賣出
            sell_price_col = self.choice_price_col(self._position, "buy")
            sell_price = sell_price_col[self._current_tick]

            self._total_profit *= (1-self.handing_charge) #買手續費
            self._total_profit = (self._total_profit / buy_price) * sell_price
            self._total_profit *= (1-self.handing_charge) #賣手續費

        # 結束
        if self._truncated:
          if action != Actions.Buy_NTD.value:
            buy_price_col = self.choice_price_col(Positions(action), "sell")
            buy_price = buy_price_col[self._last_trade_tick]


            sell_price_col = self.choice_price_col(Positions(action), "buy")
            sell_price = sell_price_col[self._current_tick]

            self._total_profit *= (1-self.handing_charge) #買手續費
            self._total_profit = (self._total_profit / buy_price) * sell_price
            self._total_profit *= (1-self.handing_charge-self.transaction_tex) #賣手續費

    def get_total_profit(self):
      return self._total_profit
  
####################################################################

torch.manual_seed(1234)
np.random.seed(1234)

def load_stock_data(csv_file)->list:
    df = pd.read_csv(csv_file)
    stock_list = ['A','B','C','D','E','F','G','H']
    price_list = []
    for stock_id in stock_list:
        price_list.append(df[df['stock_id']==stock_id])

    return price_list


origin_df_list = load_stock_data('./data/train_data.csv')


for df in origin_df_list:
    df['bar'] = df['close']/df['open'] - 1
    df['high-low'] = df['high']/df['low'] - 1
    df['high-close'] = df['close']/df['high'] - 1
    df['low-close'] = df['close']/df['low'] - 1
    

def my_calculate_reward(self, action):
    """
    進階的 reward 設計範例
    """
    # 初始化 step_reward
    step_reward = 0

    # 當持有台幣時 (未進行投資)
    if self._position == Positions.NTD:
        # 沒有收益或損失，獎勵設為 0
        step_reward = 0

    else:
        # 取得目前持有資產的價格
        price_col = self.choice_price_col(self._position)
        current_price = price_col[self._current_tick]
        last_day_price = price_col[self._current_tick - 1]

        # 計算當日收益率
        daily_return = np.log(current_price/last_day_price)

        # *** 1. 基本收益獎勵 ***
        # 如果當日收益率為正，給予正向獎勵；否則給予負向獎勵
        # step_reward = daily_return
        if daily_return >= 0:
            step_reward = daily_return
        else:
            step_reward = -(np.exp(np.abs(daily_return*5))-1)/5
        
        # *** 2. 風險控制獎勵 ***
        # 如果交易過於頻繁，給予懲罰 (例如每次改變持倉都會有小的懲罰)
        if action != self._position.value:
            step_reward -= (self.handing_charge  + self.transaction_tex) 

    return step_reward

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]

    prices_list, signal_features_list = [], []

    # used_columns = [
    #   "close","open","high","low","volume","ht_dcperiod","ht_dcphase","inphase","quadrature","sine","leadsine","adx","adxr","apo","aroondown","aroonup","aroonosc","bop","cci","cmo","dx","macd_x","macdsignal_x","macdhist_x","macd_y","macdsignal_y","macdhist_y","macd","macdsignal","macdhist","mfi","minus_di","minus_dm","mom","plus_di","plus_dm","ppo","roc","rocp","rocr","rocr100","rsi","slowk","slowd","fastk_x","fastd_x","fastk_y","fastd_y","trix","ultosc","willr","upperband","middleband","lowerband","dema","ema","ht_trendline","kama","ma","mama","fama","midpoint","midprice","sar","sarext","sma","t3","tema","trima","wma","avgprice","medprice","typprice","wclprice","beta","correl","linearreg","linearreg_angle","linearreg_intercept","linearreg_slope","stddev","tsf","var","atr","natr","trange","ad","adosc","obv"
    # ]
    used_columns = [ "close","open","high","low","volume","bar","beta",'high-low','high-close','low-close']
    # used_columns = [ "close","open","high","low","volume","bar",'high-low','high-close','low-close','return_1','return_5','sharp_10','MA_5','MA_10']
    for df in env.df_list:
      prices_list.append(df.iloc[start:end, :].filter(['open','close']))
      # 這邊可修改想要使用的 feature
      # signal_features_list.append(df.iloc[:,2:].to_numpy()[start:end])
      signal_features_list.append(df.filter(used_columns).to_numpy()[start:end])

    return prices_list, signal_features_list

class MyStockEnv(StockEnv):
    # 除 _process_data 和 _calculate_reward 外，其餘功能 (class function) 禁止覆寫
    _process_data = my_process_data
    _calculate_reward = my_calculate_reward

# window_size: 能夠看到幾天的資料當作輸入, frame_bound: 想要使用的資料日期區間
# 可修改 frame_bound 來學習不同的環境資料，frame_bound起始值必須>=window_size
# 不可修改此處 window_size 參數 ，最後計算分數,時 window_size 也會設為10
env = MyStockEnv(origin_df_list, window_size=10, frame_bound=(10, 1800))




############################################################################3
class PolicyGradientNetwork(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim*8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)


    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        state = state.flatten()
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def load_ckpt(self, ckpt_path):
      if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      else:
        print("Checkpoint file not found, use default settings")

    def save_ckpt(self, ckpt_path):
      torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_path)

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, log_probs, rewards):
        log_probs = log_probs.to(device)
        rewards = rewards.to(device)
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        state = state.flatten()
        state = torch.FloatTensor(state).to(device)
        action_prob = self.network(state)  # 將狀態轉移到 GPU
        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob



    def load_ckpt(self, ckpt_path):
      if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      else:
        print("Checkpoint file not found, use default settings")

    def save_ckpt(self, ckpt_path):
      torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_path)


network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
agent = PolicyGradientAgent(network)

EPISODE_PER_BATCH = 10  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 200     # 總共更新 200 次
CHECKPOINT_PATH = './model.ckpt' # agent model 儲存位置

avg_total_rewards = []

agent.network.train()  # 訓練前，先確保 network 處在 training 模式

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:
    log_probs, rewards = [], []
    total_rewards = []
    profit_list = []
    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):
        observation, _ = env.reset()
        total_step = 0
        while True:
            action, log_prob = agent.sample(observation)
            observation, reward, done, info = env.step(action)
            profit_list.append(info['total_profit'])
            log_probs.append(log_prob)
            total_step += 1
            if done:
              total_rewards.append(info['total_reward'])
              # 設定同一個 episode 每個 action 的 reward 都是 total reward，這邊可以更改你們覺得合理的Cumulated rewards方式
              rewards.append(np.full(total_step, info['total_reward']))
              break

    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_total_rewards.append(avg_total_reward)
    prg_bar.set_description(f"Average Reward: {avg_total_reward: 4.2f}, Final Reward: {info['total_reward']: 4.2f}, Final Profit: {info['total_profit']: 4.2f}")

    # 更新網路
    rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

# 儲存 agent model 參數
agent.save_ckpt(CHECKPOINT_PATH)


test_df = load_stock_data('./data/test_data.csv')
for df in test_df:
    df['bar'] = df['close']/df['open'] - 1
    df['high-low'] = df['high']/df['low'] - 1
    df['high-close'] = df['close']/df['high'] - 1
    df['low-close'] = df['close']/df['low'] - 1
    # df['return_1'] = df['close'].shift(1)/df['close'] - 1
    # df['return_5'] = df['close'].shift(5)/df['close'] - 1
    # df['sharp_10'] = df['return_1'].rolling(window=9).apply(
    # lambda x: (x.mean()) / x.std() if x.std() != 0 else 0,raw=True)
    # df['MA_5'] = df['close']/df['close'].rolling(5).mean() -1
    # df['MA_10'] = df['close']/df['close'].rolling(10).mean() -1
    
env = MyStockEnv(df_list=test_df, window_size=10, frame_bound=(10, 200))


network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
test_agent = PolicyGradientAgent(network)

checkpoint_path = './model.ckpt'

test_agent.load_ckpt(checkpoint_path)
test_agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

observation,_ = env.reset()
profit_list = []
action_list = []
while True:
    action, _ = test_agent.sample(observation)
    action_list.append(action)
    observation, reward, done, info = env.step(action)
    profit_list.append(info['total_profit'])
    if done:
      break

private_df = load_stock_data('./data/private_test_data.csv')

for df in private_df:
    df['bar'] = df['close']/df['open'] - 1
    df['high-low'] = df['high']/df['low'] - 1
    df['high-close'] = df['close']/df['high'] - 1
    df['low-close'] = df['close']/df['low'] - 1


frame_bounds = [(10,50), (10,100), (10,200)]
mean_profit = 0

for frame_bound in frame_bounds:
  env =  MyStockEnv(df_list=private_df, window_size=10, frame_bound=frame_bound)
  # env.reset()
  # env.render_all()

  network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
  test_agent = PolicyGradientAgent(network)

  checkpoint_path = './model.ckpt'

  test_agent.load_ckpt(checkpoint_path)
  test_agent.network.eval()

  # 我們會跑10次取平均
  for i in range(10):
    observation,_ = env.reset()
    while True:
        action, _ = test_agent.sample(observation)
        observation, reward, done, info = env.step(action)
        if done:
          break

    # env.render_all()
    # plt.show()

    mean_profit += env.get_total_profit()

mean_profit /= (10 * len(frame_bounds))
print("Score:", mean_profit)

