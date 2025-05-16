import pandas as pd
import numpy as np
import pandas_ta as ta
import ephem # type: ignore
import warnings
import uuid
import os
warnings.filterwarnings('ignore')
from untrade.client import Client
from tqdm import tqdm
class Variable_Moving_Average:
    def __init__(self, data, vma_length = 9):
        self.df = data
        self.vma_length = vma_length # length for vma 
        self.k = 1.0 / self.vma_length # scaling parameter

        # Initialize variables
        self.vma_values, self.vma_high_values, self.vma_low_values = [0.0], [0.0], [0.0]
        self.isss = [0.0]  # Store iS values for precomputing HHV and LLV
        self.rolling_hhv, self.rolling_llv = pd.Series(), pd.Series()

        # Calculate iS and VMA
        self.calculate_iS()
        self.calculate_vma()

        # Add values of vma to DataFrame
        self.df['vma'] = self.vma_values
        self.df['vma_high'] = self.vma_high_values
        self.df['vma_low'] = self.vma_low_values

    def calculate_iS(self):
        pdmS, mdmS, pdiS, mdiS, iS = 0.0, 0.0, 0.0, 0.0, 0.0 # intiailising values for calcluation

        for i in tqdm(range(1, len(self.df)),desc='Processing'):
            vmasrc = self.df['close'].iloc[i]
            prev_vmasrc = self.df['close'].iloc[i - 1]

            # Positive and negative directional movements
            pdm = max(vmasrc - prev_vmasrc, 0)
            mdm = max(prev_vmasrc - vmasrc, 0)

            # Smoothed positive and negative directional movement sums
            pdmS = (1 - self.k) * pdmS + self.k * pdm
            mdmS = (1 - self.k) * mdmS + self.k * mdm

            # Sum of smoothed movements
            s = pdmS + mdmS
            pdi = pdmS / s if s != 0 else 0
            mdi = mdmS / s if s != 0 else 0

            # Smoothed positive and negative directional indicators
            pdiS = (1 - self.k) * pdiS + self.k * pdi
            mdiS = (1 - self.k) * mdiS + self.k * mdi

            # Trend intensity
            d = abs(pdiS - mdiS)
            s1 = pdiS + mdiS
            iS = (1 - self.k) * iS + self.k * (d / s1) if s1 != 0 else 0
            self.isss.append(iS)

        # Precompute rolling max/min for iS values
        iss_series = pd.Series(self.isss)
        self.rolling_hhv = iss_series.rolling(window=self.vma_length, min_periods=1).max()
        self.rolling_llv = iss_series.rolling(window=self.vma_length, min_periods=1).min()

    def calculate_vma(self):
        vma, vma_high, vma_low = 0.0, 0.0, 0.0

        for i in tqdm(range(1, len(self.df)),desc='Calculating VMA'):
            vmasrc = self.df['close'].iloc[i]
            high_src = self.df['high'].iloc[i]
            low_src = self.df['low'].iloc[i]

            # Fetch precomputed HHV and LLV
            hhv = self.rolling_hhv.iloc[i]
            llv = self.rolling_llv.iloc[i]

            # High/low normalization and VMA calculation
            d1 = hhv - llv if hhv != llv else 1e-10  # Avoid division by zero
            vI = (self.isss[i] - llv) / d1
            vma = (1 - self.k * vI) * vma + self.k * vI * vmasrc
            vma_high = (1 - self.k * vI) * vma_high + self.k * vI * high_src
            vma_low = (1 - self.k * vI) * vma_low + self.k * vI * low_src

            # Store results
            self.vma_values.append(vma)
            self.vma_high_values.append(vma_high)
            self.vma_low_values.append(vma_low)


class Trend_Using_SuperTrend: #calculate trend using supertrend
    def __init__(self, data, multiplier):
        self.data = data
        self.multiplier = multiplier # multiplier for atr
        self.data['supertrend'] = 0.0
        self.data['supertrend_direction'] = 0.0
    
    def calculate_atr(self, high, low, close, period):
        self.data['ATR'] = ta.atr(high = high, low = low, close = close, length = period) # using pandas_ta to calculate ATR for any arbitrary hcl

    def calculate_supertrend(self, high, low, close, period):
        self.calculate_atr(high = high, low = low, close = close, period = period)
        self.data['UpperBand'] = (high + low) / 2 + self.multiplier * self.data['ATR'] # Lower Band for Supertrend
        self.data['LowerBand'] = (high + low) / 2 - self.multiplier * self.data['ATR'] # Upper Band for Supertrend

        # Generating the supertrend
        for i in tqdm(range(1, len(self.data)),desc='Calculating Supertrend'):
            # If close is above upperband, lowerband becomes supertrend
            if self.data['close'].iloc[i] > self.data['UpperBand'].iloc[i - 1]:
                self.data['supertrend'].iloc[i] = self.data['LowerBand'].iloc[i] 
            # If close is below upperband, upperband becomes supertrend
            elif self.data['close'].iloc[i] < self.data['LowerBand'].iloc[i - 1]:
                self.data['supertrend'].iloc[i] = self.data['UpperBand'].iloc[i]
            else:
            # it takes previous value
                self.data['supertrend'].iloc[i] = self.data['supertrend'].iloc[i - 1] 

    def trend_using_supertrend(self):
        # Calculating the trend of given close wrt to the given value
        for i in tqdm(range(len(self.data)),desc='Calculating Trend'):
            # Supertrend calculation using vma
            if self.data['vma'].iloc[i] > self.data['supertrend'].iloc[i]:
                self.data['supertrend_direction'].iloc[i] = 1
            elif self.data['vma'].iloc[i] < self.data['supertrend'].iloc[i]:
                self.data['supertrend_direction'].iloc[i] = -1

        self.data['trend_supertrend'] = self.data['supertrend_direction']


def Add_Trend_Lines(data): #add multiple window based trend lines
    data["trend_mean_5"]=data['trend_supertrend'].rolling(window=5).mean()
    data["trend_mean_50"]=data["trend_supertrend"].rolling(window=25).mean()
    data['trend_mean_50'] = data['trend_supertrend'].rolling(window=50).mean()
    data["trend_mean_75"]=data["trend_supertrend"].rolling(window=75).mean()
    data["trend_mean_100"]=data['trend_supertrend'].rolling(window=100).mean()
    data['trend_mean_200'] = data['trend_supertrend'].rolling(window=200).mean()
    data['close_low'] = data['close'].rolling(window=6).min()
    data['close_high'] = data['close'].rolling(window=6).max()


class Dynamic_Volatilty_Tracking: #Dyanmically assigning volatity at a given time based on previous data
    #types specified for instant resolution
    def __init__(self, df: pd.DataFrame, atr_length: int = 10, factor: float = 3, window_size: int = 500, 
                 high_vol_percentile: float = 0.75, mid_vol_percentile: float = 0.50, low_vol_percentile: float = 0.25):
        self.data = df
        self.atr_length = atr_length #length of atr of close to be chosen
        self.factor = factor
        self.window_size = window_size #volatility clusterss retraining frequency
        #percentiles for each regime
        self.high_vol_percentile, self.mid_vol_percentile, self.low_vol_percentile = high_vol_percentile, mid_vol_percentile, low_vol_percentile
        self.result_df = self.work()

    #calculating atr traditionally
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, length: int):
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr.max(axis=1)
        return tr['tr'].rolling(length).mean()


    def allocate_volatility_clusters(self, volatility: pd.Series, high_vol: float, mid_vol: float, low_vol: float, max_iter: int = 50) -> list:
        vol_range = volatility.max() - volatility.min() #range of volaility

        #defining clusters by defining thresholds
        clusters = [
            volatility.min() + vol_range * high_vol,
            volatility.min() + vol_range * mid_vol,
            volatility.min() + vol_range * low_vol
        ]

        prev_clusters = None
        current_clusters = clusters
        iterations = max_iter # number of iterations to run

        # clusters get redefined until iterations are finished or improvement stops happens
        
        while prev_clusters != current_clusters and iterations > 0:
            prev_clusters = current_clusters.copy()
            
            # Assign points to clusters
            distances = pd.DataFrame({
                'high': abs(volatility - current_clusters[0]),
                'mid': abs(volatility - current_clusters[1]),
                'low': abs(volatility - current_clusters[2])
            })

            #choose the clusters with minimum distance
            clusters = distances.idxmin(axis=1)
            
            # Update clusters
            for label, clusters_idata in [('high', 0), ('mid', 1), ('low', 2)]:
                if len(clusters[clusters == label]) > 0:
                    current_clusters[clusters_idata] = volatility[clusters == label].mean()
            
            iterations -= 1
            
        return current_clusters
    

    def classify_point(self, value: float, clusterss: list) -> int:
        # Classify a single point based on nearest clusters.
        distances = [abs(value - c) for c in clusterss]
        return distances.index(min(distances)) 
    
    
    def work(self):  # the function which through which the model works
        self.data['atr'] = self.calculate_atr(self.data['high'], self.data['low'], self.data['close'], self.atr_length)

        self.data['volatility_cluster'] = pd.NA # volatility cluster
        self.data['assigned_centroid'] = pd.NA # assigned centroid

        if len(self.data) < self.window_size:
            print("Please input bigger data")
            return self.data
        
        for i in tqdm(range(self.window_size, len(self.data)),desc='Volatility Clustering'):

            # Get ATR values till that index
            slice = self.data['atr'].iloc[:i]
            
            # allocate clusters
            if i % self.window_size == 0:
                clusters = self.allocate_volatility_clusters(
                    slice,
                    self.high_vol_percentile,
                    self.mid_vol_percentile,
                    self.low_vol_percentile
                )
            
            # Classify current point
            current_value = self.data['atr'].iloc[i]
            cluster = self.classify_point(current_value, clusters)
            
            # Store the clusters and the cluster assigned
            self.data.iloc[i, self.data.columns.get_loc('volatility_cluster')] = cluster
            self.data.iloc[i, self.data.columns.get_loc('assigned_centroid')] = clusters[cluster]

        # store the resultant dataframe 
        result_df = self.data[['datetime', 'close', 'supertrend', 'supertrend_direction', 'volatility_cluster']]

        return result_df


def heikin_ashi_candlesticks(df: pd.DataFrame):
    # Calculate HA close
    df["HA_CLOSE"] = (df["low"] + df["high"] + df["close"] + df["open"]) / 4

    for i in tqdm(range(len(df)),desc='Heiken Aishi'):
            if i == 0:
                # For the first row
                df.at[i, "HA_OPEN"] = (df.at[i, "open"] + df.at[i, "close"]) / 2
            else:
                # For subsequent rows
                df.at[i, "HA_OPEN"] = (df.at[i - 1, "HA_OPEN"] + df.at[i - 1, "HA_CLOSE"]) / 2

    # Calculate HA HIGH and LOW
    df["HA_HIGH"] = df[["HA_OPEN", "HA_CLOSE", "high"]].max(axis=1)
    df["HA_LOW"] = df[["HA_OPEN", "HA_CLOSE", "low"]].min(axis=1)

    return df


def ATR(data, high, low, close, period):  # Calculate ATR
    data['ATR'] = ta.atr(high, low, close, length=period)

## Setting up the framework for our strategy
class Portfolio(): #used to keep track of portfolio of a strategy
    def __init__(self, data,signals):
        self.data = data
        self.close = data['close']
        self.signal = signals

        self.capital = 1000 
        self.initial_capital = 1000 # start with the capital same as untrade
        self.stocks = 0 # the amount of stocks held at any given time

        #parameters for the portfolio
        self.capital_invested = 0
        self.current = 0 #current trade
        self.exit = []
        self.entry = []
        self.portfolio_value = []
        #final cleaned signal for the strategy
        self.final_signal = []


    def check_exit_condition(self,i):
        if self.current == 1: # if currently in long trade 
            if self.signal[i] == -1: # exit signal appears
              self.final_signal.append(-1) # stop the trade
              self.exit_long_position(i)
            elif self.signal[i] == -2: # reversal signal appears
                self.final_signal.append(-2) # exit the long and enter a short
                self.exit_long_position(i)
                self.enter_short_position(i)
            else:
              self.final_signal.append(0) # if no exit condition is fullfilled, do nothing

        elif self.current == -1: # if currently in short trade 
            if self.signal[i] == 1: # exit signal appears
              self.final_signal.append(1) # stop the trade
              self.exit_short_position(i)
            elif self.signal[i] == 2: # reversal signal appears
                self.final_signal.append(2) # exit the short and enter a long
                self.exit_short_position(i)
                self.enter_long_position(i)
            else:
              self.final_signal.append(0) # if no exit condition is fullfilled, do nothing


    def calculate_portfolio_value(self, i):
        # calculate portfolio value at a given index
        capital = self.capital + self.stocks*self.close[i]*self.current
        self.portfolio_value.append(capital)
        self.check_exit_condition(i)


    def enter_long_position(self,i):
        # enter a long trade and change current position to long
        if self.signal[i] == 1:
            self.capital_invested = self.capital
            self.stocks = (self.capital_invested/self.close[i])
            self.portfolio_value.append(self.capital)
            self.capital = self.capital - self.close[i]*self.stocks
            self.current = 1
        else:
            self.capital_invested = self.capital
            self.stocks = (self.capital_invested/self.close[i])
            self.capital = self.capital - self.close[i]*self.stocks
            self.current = 1

    def exit_long_position(self,i):
        #exit the long trade and change current position to 0
        self.capital = self.capital + self.stocks*self.close[i]
        self.current = 0
        self.stocks = 0

    def enter_short_position(self,i):
        # enter a short trade and change current position to short
        if self.signal[i] == -1:
          self.capital_invested = self.capital
          self.stocks = self.capital_invested/self.close[i]
          self.portfolio_value.append(self.capital)
          self.capital = self.capital + self.close[i]*self.stocks
          self.current = -1
        else:
          self.capital_invested = self.capital
          self.stocks = self.capital_invested/self.close[i]
          self.capital = self.capital + self.close[i]*self.stocks
          self.current = -1


    def exit_short_position(self,i):
        # exit the short trade and change current position to 0
        self.capital = self.capital - self.stocks*self.close[i]
        self.current = 0
        self.stocks = 0


    def backtest(self):

        for i in tqdm(range(len(self.data)-1),desc='Final Signal Generation'):
            # if there is no position
            if self.current == 0:
                if self.signal[i] == 1: # buy signal 
                    self.final_signal.append(1) # buy
                    self.enter_long_position(i)
                elif self.signal[i] == -1: # sell signal
                    self.final_signal.append(-1) # sell
                    self.enter_short_position(i)
                else:
                    self.final_signal.append(0) # if nothing happens, do nothing
                    self.calculate_portfolio_value(i)

            elif self.current != 0:
                self.calculate_portfolio_value(i) # calculate portfolio value

            # if portfolio gets wiped out, notify it
            if (self.portfolio_value[-1] < 0):
                print("Portfolio Wiped Out")
                break

        self.calculate_portfolio_value(i+1) # calculate portfolio value

        if len(self.final_signal) <len(self.data):
            self.final_signal.append(0) 

        return self.portfolio_value, self.final_signal

class RiskManagement(): # Class to Apply RiskManagement
    def __init__(self, data,signals):
        self.data = data
        self.trend_mean_50 = data['trend_mean_50']
        self.close = data['close']
        self.low = data['low']
        self.close_low = data['close_low']
        self.close_high = data['close_high']
        self.signal = signals
        self.ATR = data['ATR']
        self.data["sideways_mean"]=self.data["sideways"].rolling(14).mean() # taking rolling mean of sideways value, to remove the discreetness
        self.final_signal = [] # list of final signals

        # capital
        self.capital = 1000
        self.initial_capital = 1000
        self.stocks = 0

        # stop loss parameters
        self.stop_loss_percent_long = 0.4
        self.stop_loss_percent_short = 0.3
        self.dynamic_exit_percent = 10

        # take profit and count of how many times they were hit
        self.stop_loss = 0
        self.take_profit_long = 0
        self.take_profit_percent_long = 100
        self.short_exit_count = 0
        self.long_exit_count = 0

        # jump stop loss parameters
        self.current_maxima = 0
        self.current_minima = 0
        self.jump_percent_long = 6
        self.jump_percent_short = 3
        self.fall_percent_short = 3
        self.fall_percent_long=2
        self.multi_long = 12
        self.multi_short = 15
        self.atr_stop_loss = []

        # flag
        self.capital_invested = 0
        self.flag = 0

        # keeping count of every trade
        self.current = 0
        self.won = 0
        self.loss = 0
        self.exit = []
        self.entry = []
        self.p_and_l = []
        self.portfolio_value = []
        self.type_of_trade = []
        
        self.invested_amount = 0
        
        # holidays for stoploss parameters
        self.holidays = ["01-01","02-14","03-08","04-22","05-01","08-31","12-25","04-01"]


    def update_risk_long(self,i):
      #updating the long stoploss based on atr
      self.final_signal.append(0)
      self.current_maxima = self.capital
      self.stop_loss = (1 - self.stop_loss_percent_long)*self.close[i]
      self.atr_stop_loss = self.close_low[i] - self.ATR[i]*self.multi_long



    def update_risk_short(self,i):
        # update the risk stoploss
      self.stop_loss=(1+self.stop_loss_percent_short)*self.close[i]
      self.final_signal.append(0)



    def update_take_profits_short(self,i):
      # update take profits 
      self.final_signal.append(0)
      multiplier=(self.current_minima-self.capital)/self.current_minima
      self.current_minima=self.capital
      self.dynamic_exit=(1-self.dynamic_exit_percent)*multiplier*self.close[i]
      self.atr_dynamic_exit = self.close_low[i] - self.ATR[i]*self.multi_short




    def check_risk_hit_long(self,i):
        # check if long risk will hit
      self.jump = (self.close[i] - self.close[i-1])*100/self.close[i-1]
      self.fall = (-self.close[i] + self.close[i-1])*100/self.close[i-1]
      if (self.close[i] < self.stop_loss or self.close[i] < self.atr_stop_loss or self.jump > self.jump_percent_long or self.close[i] > self.take_profit_long or self.fall >= self.fall_percent_long):
        self.exit_long_position(i)
        p_and_l = (self.portfolio_value[i] - self.invested_amount)/self.invested_amount
        
        self.p_and_l.append((p_and_l))
        self.final_signal.append(-1)

      elif self.capital > self.current_maxima:
        self.update_risk_long(i)

      else:
        self.final_signal.append(0)
      self.capital = self.capital - self.stocks*self.close[i]



    def check_risk_hit_short(self,i):
        # check if short risk will hit
      self.jump = (self.close[i-1] - self.close[i])*100/self.close[i-1]
      self.fall = (self.capital_invested - self.capital)*100/self.capital_invested

      if (self.close[i] < self.dynamic_exit or self.close[i] < self.atr_dynamic_exit or self.jump > self.jump_percent_short or self.fall >= self.fall_percent_short or self.close[i] > self.stop_loss):
        self.exit_short_position(i)
        p_and_l = (self.portfolio_value[i] - self.invested_amount)/self.invested_amount
        self.p_and_l.append((p_and_l))
        self.final_signal.append(1)

      elif self.capital < self.current_minima:
        self.update_take_profits_short(i)

      elif(self.capital > self.current_minima):
        self.update_risk_short(i)

      else:
        self.final_signal.append(0)
      self.capital = self.capital + self.stocks*self.close[i]




    def check_exit_condition(self,i):
        # check exit condition
        if self.current == 1: 
          if self.signal[i] == 1 or self.signal[i] == 0: # if in long trade, check if risk long is hit
            self.check_risk_hit_long(i)

          elif self.signal[i] == -1: # if in long trade, exit signal
           
            self.exit_long_position(i)
            p_and_l = (self.portfolio_value[i] - self.invested_amount)/self.invested_amount
            
            self.p_and_l.append((p_and_l))
            self.final_signal.append(-1)



        elif self.current == -1:
          
          if self.signal[i] == -1 or self.signal[i] == 0: # if in short trade, check if risk short is hit
            self.check_risk_hit_short(i)

          elif self.signal[i] == 1:
           
            self.exit_short_position(i)
            p_and_l = (self.portfolio_value[i] - self.invested_amount)/self.invested_amount
            self.p_and_l.append((p_and_l))
            self.final_signal.append(1)

    # calculate portfolio value at that particular value
    def calculate_portfolio_value(self, i):

        if self.current == 1 or self.current == -1:
            self.capital = self.capital + self.stocks*self.close[i]*self.current
            self.portfolio_value.append(self.capital)
            self.check_exit_condition(i)

        elif self.current == 0 or self.current == 100:
            self.portfolio_value.append(self.capital)

    # enter a long position 
    def enter_long_position(self,i):

        self.capital_invested = self.capital
        self.stocks = (self.capital_invested/self.close[i])
        self.stop_loss = (1 - self.stop_loss_percent_long)*self.close[i]
        self.take_profit_long = ( 1 + self.take_profit_percent_long)*self.close[i]
        self.atr_stop_loss = self.close_low[i] - self.ATR[i]*self.multi_long
        self.current_maxima = self.capital
        self.capital = self.capital - self.close[i]*self.stocks
        self.current = 1
        self.entry.append(i)
        self.type_of_trade.append("long")


    # exit long position 
    def exit_long_position(self,i):

        self.current = 0
        self.stocks = 0
        self.exit.append(i)
        if (self.portfolio_value[self.exit[-1]] > self.portfolio_value[self.entry[-1]]):
            self.won += 1
        else:
            self.loss += 1
        self.p_and_l.append(self.portfolio_value[self.exit[-1]] - self.portfolio_value[self.entry[-1]])


    # enter a short position 
    def enter_short_position(self,i):

        self.capital_invested = self.capital
        self.stocks = (self.capital_invested/self.close[i])
        self.stop_loss = (1 + self.stop_loss_percent_short)*self.close[i]
        self.dynamic_exit=(1 - self.dynamic_exit_percent)*self.close[i]
        self.atr_dynamic_exit = self.close_low[i] - self.ATR[i]*self.multi_short
        self.current_minima = self.capital
        self.capital = self.capital + self.close[i]*self.stocks

        self.current = -1
        self.entry.append(i)
        self.type_of_trade.append('short')


    # exit a short position 
    def exit_short_position(self,i):

        self.current = 0
        self.stocks = 0
        self.exit.append(i)
        if (self.portfolio_value[self.exit[-1]] > self.portfolio_value[self.entry[-1]]):
            self.won += 1
        else:
            self.loss += 1
        self.p_and_l.append(self.portfolio_value[self.exit[-1]] - self.portfolio_value[self.entry[-1]])
    
    # check if holiday is there
    def is_holiday(self, i):
       date = str(self.data.datetime.iloc[i]).split(" ")[0]
       mm_dd = str(date)[5:]
       
       
       if (mm_dd) in self.holidays:
          
          return True
       else:
          return False
       
    #check the day
    def is_day(self, i):
      time = str(self.data.datetime.iloc[i]).split(" ")[1]
      hour = int(time.split(":")[0])
      if (hour<=15 and hour >= 8):
        return True
      return False

    # set parameters for the trade 
    def set_parameters(self,i):
          
      if not self.is_holiday(i):
          
        if self.data['sideways'][i] == 1:

                  self.stop_loss_percent_long = 0.05
                  self.stop_loss_percent_short = 0.05

                  self.dynamic_exit_percent = 0.05
                  self.fall_percent_long = 10

                  self.jump_percent_long = 12
                  self.jump_percent_short = 3
                  self.fall_percent_short = 2

                  self.multi_long = 4

                  self.multi_short = 4          
        else:
          
          self.stop_loss_percent_long = 0.2
          self.stop_loss_percent_short = 0.2
          self.dynamic_exit_percent = 0.2
          self.fall_percent_long = 10
        
          
          self.jump_percent_long = 12
          self.jump_percent_short = 3
          self.fall_percent_short = 2
  
          self.multi_long = 12 
          
          self.multi_short = 4
      else:
          self.stop_loss_percent_long = 0.25
          self.stop_loss_percent_short = 0.25
          self.dynamic_exit_percent = 0.25
          self.fall_percent_long = 10
          
          
          self.jump_percent_long = 12
          self.jump_percent_short = 3
          self.fall_percent_short = 2
  
          self.multi_long = 12
          
          self.multi_short = 4      

    # now add the stoploss to the strategy 
    def backtest(self):

        for i in range(len(self.data)):

            self.set_parameters(i)  # set parameters

            if self.current == 100 and self.flag == 0:
                if self.trend_mean_50[i] == 1: 
                  self.current = 0
                else:
                  self.final_signal.append(0)
                  self.calculate_portfolio_value(i)

            if self.current == 0: # if in no trade 

                if self.signal[i] == 1 and self.current == 0: # if long signal appears
                  
                  self.final_signal.append(1)
                  self.portfolio_value.append(self.capital)
                  self.invested_amount = self.capital
                  self.enter_long_position(i) # enter a long trade

                elif self.signal[i] == -1: # else if a short signal appears 
                  
                  self.final_signal.append(-1)
                  self.portfolio_value.append(self.capital)
                  self.invested_amount = self.capital
                  self.enter_short_position(i) # enter a short trade

                else:
                  self.final_signal.append(0)
                  self.calculate_portfolio_value(i)

            elif self.current != 0 and self.current != 100:
                if self.current == 1 or self.current == -1:
                    self.calculate_portfolio_value(i) # calculate portfolio value

            if (self.portfolio_value[-1] < 0):
                print(i)
                print("Portfolio Wiped Out")
            self.flag = 0

       
        return self.portfolio_value, self.final_signal

class AlternatingFramework(): # apply the alternating framework on the combination of trade types
    def __init__(self, data,signals1,signals2,signals3,portfolio_1,portfolio_2, portfolio_3,trade_type_1, trade_type_2, trade_type_3,window):
        self.data = data
        self.close = data['close']
        self.signal1 = signals1.copy()
        self.signal2 = signals2.copy()
        self.signal3 = signals3.copy()
        self.signal = self.signal1

        # use the portfolios generated by the risk management class
        self.portfolio1 = portfolio_1 
        self.portfolio2 = portfolio_2
        self.portfolio3 = portfolio_3
        
        # use trade_type columns generated by trade type class
        self.trade_type_1 = trade_type_1
        self.trade_type_2 = trade_type_2
        self.trade_type_3 = trade_type_3

        self.capital = 1000
        self.initial_capital = 1000 # initial capital
        self.stocks = 0
        
        self.strategy = []
        self.strat = 'strategy 1' # we start with strategy 1 abritrarily

        self.trend_mean_100 = self.data['trend_mean_100']
        self.rolling_sideways = self.data['sideways'].rolling(window=24).mean()

        # parameters
        self.window_bull = 0
        self.window_bear = 0
        self.window = window
        self.profit_1 = 0
        self.profit_2 = 0
        self.profit_3 = 0

        #keeeping track of metrics
        self.capital_invested = 0
        self.current = 0
        self.exit = []
        self.entry = []
        self.portfolio_value = []
        self.final_signal = []
        
        self.p_and_l = []
        self.invested_amount = 0

    # check if exit condition has been hit
    def check_exit_condition(self,i):
        if self.current == 1:
            if self.signal[i] == -1: # exit a long trade
                self.exit_long_position(i)
                self.exit.append(i)
                self.final_signal.append(-1)
                p_and_l = (self.portfolio_value[-1] - self.invested_amount)/self.invested_amount
                
                self.p_and_l.append(p_and_l)
            elif self.signal[i] == -2: # reversal
                
                p_and_l = (self.portfolio_value[-1] - self.invested_amount)/self.invested_amount
                self.p_and_l.append((p_and_l))
                self.exit_long_position(i)
                self.enter_short_position(i)
                
                self.invested_amount = self.portfolio_value[-1]
                self.exit.append(i)
                self.entry.append(i)
                self.final_signal.append(-2)
            else:
                self.final_signal.append(0)

        elif self.current == -1: # exit a short trade
            if self.signal[i] == 1:
                self.exit_short_position(i)
                p_and_l = (self.portfolio_value[-1] - self.invested_amount)/self.invested_amount
                self.p_and_l.append((p_and_l))
            
                self.final_signal.append(1) # go to trade 0
                self.exit.append(i)
            elif self.signal[i] == 2: # bullish reversal 
                p_and_l = (self.portfolio_value[-1] - self.invested_amount)/self.invested_amount
                self.p_and_l.append((p_and_l))
                self.exit_short_position(i) # close the short position
                self.enter_long_position(i) # enter long trade
                
                self.invested_amount = self.portfolio_value[-1]
                self.exit.append(i)
                self.entry.append(i)
                self.final_signal.append(2)
            else:
                self.final_signal.append(0)

    # the main logic of alternating framework
    def assign_strategy(self,i):
      # if in sideways regime, we consider fibonacci as it is custom made for sideways regime
      # we seek which strategy made high profit in lookback window
      if self.rolling_sideways[i] == 1:
        max_profit = max(self.profit_1, self.profit_2, self.profit_3) 
      else:
        max_profit = max(self.profit_1, self.profit_2)

      # We move to strategy which made the highest profit in the lookback window
      if self.profit_1 == max_profit:
        self.signal = self.signal1.copy()
        self.strat = "strategy 1"
        if self.trade_type_1[i] == 'In long trade':
            self.signal[i] = 1
        elif self.trade_type_1[i] == 'In short trade':
            self.signal[i] = -1
            
      elif max_profit == self.profit_2:
        self.strat = "strategy 2"
        self.signal = self.signal2.copy()
        if self.trade_type_2[i] == 'In long trade':
            self.signal[i] = 1
        elif self.trade_type_2[i] == 'In short trade':
            self.signal[i] = -1

            
      elif max_profit == self.profit_3:
        self.strat = "strategy 3"
        self.signal = self.signal3.copy()
        if self.trade_type_3[i] == 'In long trade':
            self.signal[i] = 1
        elif self.trade_type_3[i] == 'In short trade':
            self.signal[i] = -1


    def check_strategy_switch(self,i):
        # to check profits for all strategies and switch based on profit

      if i >= self.window:
        self.profit_1 = (self.portfolio1[i-1] - self.portfolio1[i-self.window])*100/self.portfolio1[i-self.window]
        self.profit_2 = (self.portfolio2[i-1] - self.portfolio2[i-self.window])*100/self.portfolio2[i-self.window]
        self.profit_3 = (self.portfolio3[i-1] - self.portfolio3[i-self.window])*100/self.portfolio3[i-self.window]

      self.assign_strategy(i) # assign strategy


    def calculate_portfolio_value(self, i): # calculate portfolio value
        capital = self.capital + self.stocks*self.close[i]*self.current
        self.portfolio_value.append(capital)
        self.check_exit_condition(i)


    def enter_long_position(self,i):
        # enter a long position
        if self.signal[i] == 1:
            self.capital_invested = self.capital
            self.stocks = (self.capital_invested/self.close[i])
            self.portfolio_value.append(self.capital)
            self.capital = self.capital - self.close[i]*self.stocks
            self.current = 1
        else:
            self.capital_invested = self.capital
            self.stocks = (self.capital_invested/self.close[i])
            self.capital = self.capital - self.close[i]*self.stocks
            self.current = 1

    def exit_long_position(self,i):
        # exit long position
        self.capital = self.capital + self.stocks*self.close[i]
        self.current = 0
        self.stocks = 0

    def enter_short_position(self,i):
        # enter a short position
        if self.signal[i] == -1:
          self.capital_invested = self.capital
          self.stocks = self.capital_invested/self.close[i]
          self.portfolio_value.append(self.capital)
          self.capital = self.capital + self.close[i]*self.stocks
          self.current = -1
        else:
          self.capital_invested = self.capital
          self.stocks = self.capital_invested/self.close[i]
          self.capital = self.capital + self.close[i]*self.stocks
          self.current = -1


    def exit_short_position(self,i):
        # exit short trade and return to neutral position
        self.capital = self.capital - self.stocks*self.close[i]
        self.current = 0
        self.stocks = 0


    def backtest(self):

        for i in tqdm(range(len(self.data)),desc='Alternating Framework'):

            if self.current == 0:

                self.check_strategy_switch(i) # check which strategy to use

                if self.signal[i] == 1: # enter long position based on the chosen strategy
                    self.final_signal.append(1)
                    self.invested_amount = self.portfolio_value[-1]
                    self.entry.append(i)
                    self.enter_long_position(i)
                
                elif self.signal[i] == -1: # enter short position based on the chosen strategy
                    self.entry.append(i)
                    self.final_signal.append(-1)
                    self.invested_amount = self.portfolio_value[-1]
                    self.enter_short_position(i)
                
                else:
                    self.final_signal.append(0) # do nothing
                    self.calculate_portfolio_value(i)

            elif self.current != 0:
                #calculate portfolio value
                if self.current == 1:
                    self.calculate_portfolio_value(i)
                elif self.current == -1:
                    self.calculate_portfolio_value(i)

            self.strategy.append(self.strat)
            #if portfolio gets wiped out, notify so
            if (self.portfolio_value[-1] < 0):
                print("Portfolio Wiped Out")
                break
            
       
            
        return self.portfolio_value, self.final_signal, self.strategy, self.p_and_l
    

class TradeType: # add trade_type column to our dataframe
    def __init__(self, data, signals):
        self.data = data
        self.signal = signals
        self.current = 0         # 0 for no position, 1 for long, -1 for short
        self.type_of_trade = []  # List to store the type of each trade based on signal

    def determine_trade_type(self):
        for i in range(len(self.data)):
            signal = self.signal[i]

            if self.current == 1:  # In a long position
                if signal in [0, 1]:
                    self.type_of_trade.append("In long trade")
                elif signal == -1:
                    self.current = 0
                    self.type_of_trade.append("Close")
                elif signal == -2:
                    self.current = -1
                    self.type_of_trade.append("Long_reversal")

            elif self.current == -1:  # In a short position
                if signal in [0, -1]:
                    self.type_of_trade.append("In short trade")
                elif signal == 1:
                    self.type_of_trade.append("Close")
                    self.current = 0
                elif signal == 2:
                    self.type_of_trade.append("Short_reversal")
                    self.current = 1

            else:  # No position
                if signal == 1:
                    self.type_of_trade.append("Long")
                    self.current = 1
                elif signal == -1:
                    self.type_of_trade.append("Short")
                    self.current = -1
                elif signal == 0:
                    self.type_of_trade.append("No trade")
                    self.current = 0


        return self.type_of_trade

## Defining & Applying our strategy 
class strat1: #strategy based on keltner signals
    def __init__(self, data):
        self.data = data
        # set the parameters for the strategy
        ATR(self.data, self.data['HA_HIGH'], self.data['HA_LOW'], self.data['HA_CLOSE'], period = 10)
        self.data['EMA'] = self.data['close'].ewm(span=20, adjust=False).mean()
        self.data['keltner_ub'] = self.data['EMA'] + 2.25 * self.data['ATR'] # set keltner upper bound
        self.data['keltner_lb'] = self.data['EMA'] - 2.25 * self.data['ATR'] # set keltner lower bound
        self.keltner() # run the strategy

    def keltner(self):
        self.data['signal_keltner'] = 0 # initialise the signals column for the strategy

        for i in range(10, len(self.data)):
            # check for a close and keltner upperbound crossover and confirm it with sma with 75 window
            if (self.data['close'].iloc[i] > self.data['keltner_ub'].iloc[i]) and (self.data['close'].iloc[i-1] < self.data['keltner_ub'].iloc[i-1]) and self.data["trend_mean_75"].iloc[i] >= 0.7:
                self.data['signal_keltner'].iloc[i] = 1
            # check for a close and keltner lowerbound crossover and confirm it with sma with 75 window
            elif (self.data['close'].iloc[i] < self.data['keltner_lb'].iloc[i]) and (self.data['close'].iloc[i-1] > self.data['keltner_lb'].iloc[i-1]) and self.data["trend_mean_75"].iloc[i] <= -0.7:
                self.data['signal_keltner'].iloc[i] = -1


class Lunar_Modification: #applying lunar stoploss on keltner based strategy
    def __init__(self, data, threshold,lunar_factor):
        self.data = data
        self.threshold = threshold # threshold days
        self.data['datetime'] = pd.to_datetime(self.data['datetime'], errors='coerce')
        self.data['date'] = self.data['datetime'].dt.date
        self.data['time'] = self.data['datetime'].dt.time
        self.data['moon_signals'] = 0
        self.lunar_factor = lunar_factor # 72 for btc & 48 for eth
        self.generating_moon_signals() # apply the strategy

    def moon_phase_flag(self, date_str, threshold_days = 1): #calculate if today is with threshold number of days of new moon or full moon
        today_date = pd.to_datetime(date_str).date()
        next_full_moon = ephem.next_full_moon(date_str).datetime().date() # get the next full moon 
        next_new_moon = ephem.next_new_moon(date_str).datetime().date() # get the next new moon
        days_to_full_moon = abs((next_full_moon - today_date).days)
        days_to_new_moon = abs((next_new_moon - today_date).days)
        
        # check if date satisfies the threshold
        if days_to_full_moon <= threshold_days:
            return 1
        
        elif days_to_new_moon <= threshold_days:
            return -1
        
        else:
            return 0
        
        
    def generating_moon_signals(self):
        for i in tqdm(range(len(self.data)),desc='Lunar Signals'):
            today_date_str = self.data.loc[i, "date"]
            current_signal = self.moon_phase_flag(today_date_str, self.threshold) # get the signal using the threshold and ephem library

            # if there are no moon signals at 11, but there is a current signal, the current signal becomes active
            if self.data.iloc[i - 24]['moon_signals'] == 0 and str(self.data.iloc[i]['time']).split(":")[0] == '23': 
                self.data.loc[i, "moon_signals"] = current_signal
            
            # if there is a signal 24 hours before but not 48 hours before, we take the 24 hours before
            if self.data.iloc[i - 24]['moon_signals'] != 0 and self.data.iloc[i - 48]['moon_signals'] == 0 and str(self.data.iloc[i]['time']).split(":")[0] == '23':
                self.data.loc[i , "moon_signals"] = self.data.iloc[i - 24]['moon_signals']

            # if there is a signal 24 hours before and not 48 hours before, we take the current signal
            if self.data.iloc[i - 24]['moon_signals'] != 0 and self.data.iloc[i - 48]['moon_signals'] != 0 and str(self.data.iloc[i]['time']).split(":")[0] == '23':
                self.data.loc[i, "moon_signals"] = current_signal

        position, index = 0, 0

        for i in tqdm(range(47, len(self.data) - 24, 24),desc='Lunar Modification'):
            # if there is a moon signal now and 24 hours before we store the position and index
            if self.data.iloc[i]['moon_signals'] == 1 and self.data.iloc[i-24]['moon_signals'] == 1:
                position=1
                index=i
            
            if self.data.iloc[i]['moon_signals'] == -1 and self.data.iloc[i-24]['moon_signals'] == -1:
                position=-1
                index=i
                
            # if the difference equals lunar factor
            if (i - index) == self.lunar_factor and str(self.data.iloc[i]['time']).split(":")[0] == '23':
                # confirm the with close of the day before and modify the signals
                if position == 1 and self.data.iloc[i]['close'] > self.data.iloc[index]['close']:
                    position = 0
                    self.data.loc[i, 'signal_keltner'] = 1
                    self.data.loc[i + 24, 'signal_keltner'] = 1

                if position == -1 and self.data.iloc[i]['close'] < self.data.iloc[index]['close']:
                    position = 0
                    self.data.loc[i,'signal_keltner'] = -1
                    self.data.loc[i + 24, 'signal_keltner'] = -1


class strat2: #strategy based on hawkes process
        def __init__(self, data):
            self.data = data
            self.kappa, self.norm_lookback, self.signal_lookback, self.perc5, self.perc95 = 0.286, 412, 549, 0.0105, 0.979 # parameters of hawkes process
            self.data['atr'] = ta.atr(np.log(self.data['high']), np.log(self.data['low']), np.log(self.data['close']), self.norm_lookback) # atr based on log price
            self.data['norm_range'] = (np.log(self.data['high']) - np.log(self.data['low'])) / self.data['atr']
            self.alpha = np.exp(-self.kappa)
            self.arr = data['norm_range'].to_numpy()
            self.output = np.full(len(data), np.nan)
            self.generate_output_array()
            self.apply_hawkes_process()
            self.filter_signals()

        def generate_output_array(self): # using the recursive relation to generate the output array
            for i in range(1, len(self.data)):
                if np.isnan(self.output[i - 1]):
                    self.output[i] = self.arr[i]
                else:
                    self.output[i] = self.output[i - 1] * self.alpha + self.arr[i]

        def apply_hawkes_process(self):
            # applying the hawkes process 
            self.data['v_hawk'] = pd.Series(self.output, index = self.data.index) * self.kappa

            signal = np.zeros(len(self.data['close']))
            #generating percentile division in the lookback window
            q05 = self.data['v_hawk'].rolling(self.signal_lookback).quantile(self.perc5)
            q95 = self.data['v_hawk'].rolling(self.signal_lookback).quantile(self.perc95)

            last_below = -1
            curr_sig = 0


            for i in range(len(signal)):
                # v_hawk is less than the 5th percentile 
                if self.data['v_hawk'].iloc[i] < q05.iloc[i]:
                    last_below = i
                    curr_sig = 0

                if ( 
                    # We look for an excitation when it crosses the 95th percentile
                    self.data['v_hawk'].iloc[i] > q95.iloc[i]
                    and self.data['v_hawk'].iloc[i - 1] <= q95.iloc[i - 1]
                    and last_below > 0
                ):
                    change = self.data['close'].iloc[i] - self.data['close'].iloc[last_below]
                    curr_sig = 1 if change > 0.0 else -1
                signal[i] = curr_sig

            self.data['hawks_signals'] = signal

        def filter_signals(self):
            # as hawkes process captures a lot of signals, we filter them using trend mean 50
            ATR(self.data, self.data['HA_HIGH'], self.data['HA_LOW'], self.data['HA_CLOSE'], period = 10)

            self.data["hawks_signals_filter"] = np.where(
                (self.data["hawks_signals"] == 1) & (self.data["trend_mean_50"] == 1), 1,
                np.where((self.data["hawks_signals"] == -1) & (self.data["trend_mean_50"] == -1), -1, 0)
            )


class strat3: #strategy based on fibonacci levels for sideways regime
    def __init__(self, data):
        self.data = data
        self.fibonacci_levels()
        self.sideways_fibo()


    def fibonacci_levels(self):
        level1_long = []
        level2_long = []
        level3_long = []
        level4_long = []
        level1_short = []
        level2_short = []
        level3_short = []
        level4_short = []
        
        for i in range(len(self.data)):
            # we defined the fibonacci levels using the method used in the approach
            max_val = self.data['high'][i - 100 : i].max()
            min_val = self.data['low'][i - 100 : i].min()
        
        
            level1_long.append(max_val - 0.236 * (max_val - min_val))
            level2_long.append(max_val - 0.382 * (max_val - min_val))
            level3_long.append(max_val - 0.5 * (max_val - min_val))
            level4_long.append(max_val - 0.786 * (max_val - min_val))
        
            level1_short.append(min_val + 0.236 * (max_val - min_val))
            level2_short.append(min_val + 0.382 * (max_val - min_val))
            level3_short.append(min_val + 0.5 * (max_val - min_val))
            level4_short.append(min_val + 0.786 * (max_val - min_val))
        
        self.data['level1_long'] = level1_long
        self.data['level1_short'] = level1_short
        self.data['level2_long'] = level2_long
        self.data['level2_short'] = level2_short
        self.data['level3_long'] = level3_long
        self.data['level3_short'] = level3_short


    def sideways_fibo(self):
        self.data['signals_fibo'] = 0
        self.data['trade_type'] = 0
        # we used atr based on ema
        ATR(self.data, self.data['vma_high'], self.data['vma_low'], self.data['vma'], 10)
        self.data['trend_mean_24'] = self.data['trend_supertrend'].rolling(window=20).mean()
        self.data['trend_mean_60'] = self.data['trend_supertrend'].rolling(window=60).mean()
        self.data['cloud_width_mean'] = (self.data['level1_long'] - self.data['level1_short']).rolling(window=20).mean() # cloud width mean
        self.data['cloud_width'] = (self.data['level1_long'] - self.data['level1_short']) # cloud width

        for i in range(len(self.data)):
            # if cloud width is less than close / 30 and confirmed with trend_mean_24
            if self.data['cloud_width'][i] < self.data["close"][i]/30 and self.data['trend_mean_24'][i] == 1 :

                self.data['signals_fibo'][i] = 1
                
            # if cloud width is less than cloud width mean and confirmed with trend_mean_24
            elif self.data['cloud_width'][i] < self.data['cloud_width_mean'][i] and self.data['trend_mean_24'][i] == -1 :
                self.data['signals_fibo'][i] = -1



       

def process_data(data):
    #VMA Denoising the Data
    data.columns = data.columns.str.lower()
    Variable_Moving_Average(data)

    #calculating trend using supertrend using vma
    trend = Trend_Using_SuperTrend(data, 1.3) 
    trend.calculate_supertrend(data['vma_high'], data['vma_low'], data['vma'], 10)
    trend.trend_using_supertrend()   

    # adding mean trend lines of data
    Add_Trend_Lines(data)

    # add market sideways regime identification 
    volatile = Dynamic_Volatilty_Tracking(
        data,
        atr_length=10,
        factor=3,
        window_size=100,
        high_vol_percentile=0.75,
        mid_vol_percentile=0.50,
        low_vol_percentile=0.25
    )

    cluster = volatile.result_df
    cluster['volatility_cluster'] = pd.to_numeric(cluster['volatility_cluster'], errors='coerce')
    cluster_window = 24
    #defining the volatility regime to be mode of the value in previous 24 windows
    data['volatility_regime'] = (
        cluster['volatility_cluster']
        .rolling(window=cluster_window)
        .apply(lambda x: x.mode()[0] if not x.mode().empty else None, raw=False)
    )

    # defining only two regimes
    data['volatility_regime'] = data['volatility_regime'].replace(1.0, 0.0)
    data["sideways"] = np.where((data["volatility_regime"] == 2.0) | (data["volatility_regime"] == 1.0), 1, 0)
    
    return data


def strat(data, lunar_factor=48):

    data["datetime"] = pd.to_datetime(data["datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')

    

    data = heikin_ashi_candlesticks(data) # denoising the data

    strat1(data) # applying keltner strategy 
    Lunar_Modification(data, 3,lunar_factor) # modifying using lunar model
    data["portfolio 1"], data["signals 1"] = RiskManagement(data, data['signal_keltner']).backtest() # applying risk management

    strat2(data) # applying strategy based on hawkes signals 
    data["portfolio 2"], data["signals 2"] = RiskManagement(data,data['hawks_signals_filter']).backtest() # applying risk management

    strat3(data) # applying fibonacci strategy
    data["portfolio 3"], data["signals 3"] = RiskManagement(data,data['signals_fibo']).backtest()  # applying risk management

    # generating trade type column for all strategies to use in alternating framework
    keltner = TradeType(data, data['signals 1'])
    data['trade_type_1'] = keltner.determine_trade_type()
    hawks = TradeType(data, data['signals 2'])
    data['trade_type_2'] = hawks.determine_trade_type()
    fibo = TradeType(data, data['signals 3'])
    data['trade_type_3'] = fibo.determine_trade_type()

    # Using the alteranting framework
    data["alteration_portfolio"], data["alteration_signals"], data['working strategy'], profit = AlternatingFramework(data, data["signals 1"], data["signals 2"], data["signals 3"], data["portfolio 1"], data["portfolio 2"], data["portfolio 3"], data["trade_type_1"], data["trade_type_2"], data["trade_type_3"], 100).backtest()
    data["trade_type_alternating"] = TradeType(data, data['alteration_signals']).determine_trade_type() # generating the tradetype column
    data["signals"]=data["alteration_signals"] # using the alteration signals as our final signals 
    data["trade_type"]=data["trade_type_alternating"]  # using the alteration trade type as our final trade type


    return data


def perform_backtest(csv_file_path):
     client = Client()
     result = client.backtest(
         jupyter_id="team30_zelta_hpps",  # the one you use to login to jupyter.untrade.io
         file_path=csv_file_path,
         leverage=1,  # Adjust leverage as needed
     )
     return result

 # Following function can be used for every size of file, specially for large files(time consuming,depends on upload speed and file size)
def perform_backtest_large_csv(csv_file_path):
     client = Client()
     file_id = str(uuid.uuid4())
     chunk_size = 90 * 1024 * 1024
     total_size = os.path.getsize(csv_file_path)
     total_chunks = (total_size + chunk_size - 1) // chunk_size
     chunk_number = 0
     if total_size <= chunk_size:
         total_chunks = 1
        
         result = client.backtest(
             file_path=csv_file_path,
             leverage=1,
             jupyter_id="team30_zelta_hpps",
             
         )
         for value in result:
             print(value)

         return result

     with open(csv_file_path, "rb") as f:
         while True:
             chunk_data = f.read(chunk_size)
             if not chunk_data:
                 break
             chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
             with open(chunk_file_path, "wb") as chunk_file:
                 chunk_file.write(chunk_data)

             
             result = client.backtest(
                 file_path=chunk_file_path,
                 leverage=1,
                 jupyter_id="team30_zelta_hpps",
                 file_id=file_id,

                 chunk_number=chunk_number,
                 total_chunks=total_chunks,
                 
             )

             for value in result:
                 print(value)

             os.remove(chunk_file_path)

             chunk_number += 1

     return result
def main():
    # Load the data
# """
# Define Required Parameters
# """

# lunar_factor = 48
# # 48 window works for eth
# # 72 window works for btc and due to their different observed cycles.
# """"""
     data = pd.read_csv("/home/jovyan/work/data/BTC_data/BTC2019_2023_1h.csv")
     lunar_factor=72

     processed_data = process_data(data)

     result_data = strat(processed_data,lunar_factor)

     csv_file_path = "./results_btc.csv"

     result_data.to_csv(csv_file_path, index=False)

     backtest_result = perform_backtest_large_csv(csv_file_path)
    

     # No need to use following code if you are using perform_backtest_large_csv
     print(backtest_result)
     for value in backtest_result:
         print(value)


if __name__ == "__main__":
     main()











