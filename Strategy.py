import numpy as np
import pandas as pd


class BaseStrategy:
    def __init__(self, predictions, etf_data):
        """
        初始化策略基类。

        :param predictions: 预测值数据 (DataFrame)。
        :param etf_data: ETF 数据 (DataFrame)。
        """
        self.predictions = predictions
        self.etf_data = etf_data

    def generate_signals(self):
        """
        生成买卖信号（子类需要实现）。

        :return: 包含信号的 DataFrame。
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Strategy(BaseStrategy):
    def __init__(self, predictions, etf_data):
        super().__init__(predictions, etf_data)

        # 策略参数
        self.buy_threshold = None  # 买入阈值
        self.buy_time = None  # 买入时间（例如 '14:30'）
        self.sell_time = None  # 卖出时间（例如 '09:31'）

    def set_parameters(self, buy_threshold, buy_time, sell_time):
        """
        设置策略参数。

        :param buy_threshold: 买入阈值。
        :param buy_time: 买入时间（字符串格式，如 '14:30'）。
        :param sell_time: 卖出时间（字符串格式，如 '09:31'）。
        """
        self.buy_threshold = buy_threshold
        self.buy_time = buy_time
        self.sell_time = sell_time

    def generate_signals(self):
        """
        根据策略生成买卖信号。

        :return: 包含信号的 DataFrame。
        """
        signals = []

        for i in range(len(self.predictions) - 2):  # 需要至少三天数据
            date = self.predictions.iloc[i]['date']
            next_date = self.predictions.iloc[i + 1]['date']
            next_next_date = self.predictions.iloc[i + 2]['date']
            prediction = self.predictions.iloc[i]['prediction']

            if prediction > self.buy_threshold:
                buy_price = np.nan
                sell_price = np.nan
                sell_date = None

                # 当天买入
                if self.buy_time.startswith('same_day'):
                    buy_price = self._get_price(date, self.buy_time.split('_')[-1])

                    # 第二天卖出
                    if self.sell_time.startswith('next_day'):
                        sell_price = self._get_price(next_date, self.sell_time.split('_')[-1])
                        sell_date = next_date

                    # 第三天卖出
                    elif self.sell_time.startswith('next_next_day'):
                        sell_price = self._get_price(next_next_date, self.sell_time.split('_')[-1])
                        sell_date = next_next_date

                # 第二天买入
                elif self.buy_time.startswith('next_day'):
                    buy_price = self._get_price(next_date, self.buy_time.split('_')[-1])

                    # 第三天卖出
                    if self.sell_time.startswith('next_day'):
                        sell_price = self._get_price(next_next_date, self.sell_time.split('_')[-1])
                        sell_date = next_next_date

                    # 第四天卖出
                    elif self.sell_time.startswith('next_next_day'):
                        try:
                            fourth_date = self.predictions.iloc[i + 3]['date']
                            sell_price = self._get_price(fourth_date, self.sell_time.split('_')[-1])
                            sell_date = fourth_date
                        except IndexError:
                            sell_price = np.nan  # 数据不足时跳过

                # 确保买入和卖出价格有效
                if not np.isnan(buy_price) and not np.isnan(sell_price):
                    signals.append({
                        'buy_date': date,
                        'sell_date': sell_date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'prediction': prediction
                    })

        return pd.DataFrame(signals)

    def _get_price(self, date, time):
        """
        获取指定日期和时间的价格。

        :param date: 日期（字符串格式，如 '20200102'）。
        :param time: 时间（字符串格式，如 '14:30'）。
        :return: 对应时间的价格（如果不存在则返回 NaN）。
        """
        formatted_date = pd.to_datetime(date, format='%Y%m%d').strftime('%Y-%m-%d')
        timestamp = f"{formatted_date} {time}:00"

        try:
            price = self.etf_data.loc[timestamp]['close']
        except KeyError:
            price = np.nan

        return price


class RollingStrategy(BaseStrategy):
    def __init__(self, predictions, etf_data, buy_threshold, buy_time, sell_time):
        """
        初始化滚仓策略。

        :param predictions: 预测值数据 (DataFrame)。
        :param etf_data: ETF 数据 (DataFrame)。
        :param buy_threshold: 买入阈值。
        :param buy_time: 买入时间（字符串格式，如 '14:30'）。
        :param sell_time: 卖出时间（字符串格式，如 '09:31'）。
        """
        super().__init__(predictions, etf_data)
        self.buy_threshold = buy_threshold
        self.buy_time = buy_time
        self.sell_time = sell_time

    def generate_signals(self):
        """
        根据滚仓策略生成买卖信号。

        :return: 包含信号的 DataFrame。
        """
        signals = []
        holding = False  # 是否持有仓位

        for i in range(len(self.predictions) - 1):
            date = self.predictions.iloc[i]['date']
            next_date = self.predictions.iloc[i + 1]['date']
            prediction = self.predictions.iloc[i]['prediction']

            # 判断是否满足买入条件
            if prediction > self.buy_threshold and not holding:
                buy_price = self._get_price(date, self.buy_time)
                sell_price = self._get_price(next_date, self.sell_time)

                if not np.isnan(buy_price) and not np.isnan(sell_price):
                    signals.append({
                        'buy_date': date,
                        'sell_date': next_date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'prediction': prediction
                    })
                    holding = True  # 开始持仓

            # 如果已经持仓，检查是否需要卖出
            elif holding and prediction <= self.buy_threshold:
                holding = False  # 清仓

        return pd.DataFrame(signals)

    def _get_price(self, date, time):
        """
        获取指定日期和时间的价格。

        :param date: 日期（字符串格式，如 '20200102'）。
        :param time: 时间（字符串格式，如 '14:30'）。
        :return: 对应时间的价格（如果不存在则返回 NaN）。
        """
        formatted_date = pd.to_datetime(date, format='%Y%m%d').strftime('%Y-%m-%d')
        timestamp = f"{formatted_date} {time}:00"

        try:
            price = self.etf_data.loc[timestamp]['close']
        except KeyError:
            price = np.nan

        return price


class MultiFactorStrategy(BaseStrategy):
    def __init__(self, predictions, etf_data, buy_threshold, buy_time, sell_time, volume_threshold):
        """
        初始化多因子策略。

        :param predictions: 预测值数据 (DataFrame)。
        :param etf_data: ETF 数据 (DataFrame)。
        :param buy_threshold: 买入阈值。
        :param buy_time: 买入时间（字符串格式，如 '14:30'）。
        :param sell_time: 卖出时间（字符串格式，如 '09:31'）。
        :param volume_threshold: 成交量阈值。
        """
        super().__init__(predictions, etf_data)
        self.buy_threshold = buy_threshold
        self.buy_time = buy_time
        self.sell_time = sell_time
        self.volume_threshold = volume_threshold

    def generate_signals(self):
        """
        根据多因子策略生成买卖信号。

        :return: 包含信号的 DataFrame。
        """
        signals = []

        for i in range(len(self.predictions) - 1):
            date = self.predictions.iloc[i]['date']
            next_date = self.predictions.iloc[i + 1]['date']
            prediction = self.predictions.iloc[i]['prediction']

            # 判断是否满足买入条件（预测值和成交量）
            buy_volume = self._get_volume(date, self.buy_time)
            if prediction > self.buy_threshold and buy_volume > self.volume_threshold:
                buy_price = self._get_price(date, self.buy_time)
                sell_price = self._get_price(next_date, self.sell_time)

                if not np.isnan(buy_price) and not np.isnan(sell_price):
                    signals.append({
                        'buy_date': date,
                        'sell_date': next_date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'prediction': prediction
                    })

        return pd.DataFrame(signals)

    def _get_volume(self, date, time):
        """
        获取指定日期和时间的成交量。

        :param date: 日期（字符串格式，如 '20200102'）。
        :param time: 时间（字符串格式，如 '14:30'）。
        :return: 对应时间的成交量（如果不存在则返回 NaN）。
        """
        formatted_date = pd.to_datetime(date, format='%Y%m%d').strftime('%Y-%m-%d')
        timestamp = f"{formatted_date} {time}:00"

        try:
            volume = self.etf_data.loc[timestamp]['volume']
        except KeyError:
            volume = np.nan

        return volume


