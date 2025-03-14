import pandas as pd

class Backtest:
    def __init__(self, strategy, initial_capital=1e6, transaction_cost=0.001, start_date=None, end_date=None,
                 accumulation=True, intensity_deduct=False):
        """
        初始化回测类。

        :param strategy: 策略类实例。
        :param initial_capital: 初始资金。
        :param transaction_cost: 交易手续费率。
        :param start_date: 回测起始日期。
        :param end_date: 回测结束日期。
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.results = None
        self.accumulation = accumulation
        self.intensity_deduct = intensity_deduct

    def run_backtest(self):
        """
        执行回测。
        """
        signals = self.strategy.generate_signals()

        # 确保 buy_date 和 sell_date 是字符串类型
        signals['buy_date'] = signals['buy_date'].astype(str)
        signals['sell_date'] = signals['sell_date'].astype(str)

        # 根据开始和结束时间过滤信号
        if self.start_date:
            start_date_str = self.start_date.strftime('%Y%m%d')
            signals = signals[signals['buy_date'] >= start_date_str]

        if self.end_date:
            end_date_str = self.end_date.strftime('%Y%m%d')
            signals = signals[signals['sell_date'] <= end_date_str]

        if self.accumulation:
            portfolio = self._simulate_trading(signals)
        else:
            portfolio = self._simulate_trading_no_accumulation(signals)

        self.results = portfolio

    def _simulate_trading(self, signals):
        """
        模拟交易过程。

        :param signals: 买卖信号 DataFrame。
        :return: 包含每日持仓和资金变化的 DataFrame。
        """
        capital = self.initial_capital
        portfolio = []

        for _, signal in signals.iterrows():
            buy_price = signal['buy_price']
            sell_price = signal['sell_price']
            prediction = signal['prediction']

            # todo: experiment code
            if self.intensity_deduct:
                if prediction > 200 and prediction < 300:
                    # 1/4 position
                    shares = capital / buy_price * (1 - self.transaction_cost) / 4
                elif prediction >= 300 and prediction < 400:
                    # 1/2 position
                    shares = capital / buy_price * (1 - self.transaction_cost) / 2
                elif prediction >= 400 and prediction < 500:
                    # 3/4 position
                    shares = capital / buy_price * (1 - self.transaction_cost) / 1.33
                else:
                    shares = capital / buy_price * (1 - self.transaction_cost)
            else:
                # 计算买入数量
                shares = capital / buy_price * (1 - self.transaction_cost)

            pnl = shares * (sell_price - buy_price) * (1 - self.transaction_cost)

            # 更新资本
            capital += pnl

            portfolio.append({
                'buy_date': signal['buy_date'],
                'sell_date': signal['sell_date'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'pnl': pnl,
                'capital': capital
            })

        return pd.DataFrame(portfolio)

    def _simulate_trading_no_accumulation(self, signals):
        """
               模拟交易过程。

               :param signals: 买卖信号 DataFrame。
               :return: 包含每日持仓和资金变化的 DataFrame。
               """
        capital = self.initial_capital
        portfolio = []

        for _, signal in signals.iterrows():
            buy_price = signal['buy_price']
            sell_price = signal['sell_price']

            # 计算买入数量
            shares = self.initial_capital / buy_price * (1 - self.transaction_cost)
            pnl = shares * (sell_price - buy_price) * (1 - self.transaction_cost)

            # 更新资本
            capital += pnl

            portfolio.append({
                'buy_date': signal['buy_date'],
                'sell_date': signal['sell_date'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'pnl': pnl,
                'capital': capital
            })

        return pd.DataFrame(portfolio)



    def get_results(self):
        """
        获取回测结果。

        :return: 回测结果 DataFrame。
        """
        return self.results