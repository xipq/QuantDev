import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Metrics:
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
        """
        计算夏普比率。

        :param returns: 收益率序列。
        :param risk_free_rate: 无风险利率。
        :return: 夏普比率。
        """
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    @staticmethod
    def calculate_max_drawdown(capital):
        """
        计算最大回撤。

        :param capital: 资金序列。
        :return: 最大回撤。
        """
        cumulative_max = capital.cummax()
        drawdown = (cumulative_max - capital) / cumulative_max
        max_drawdown = drawdown.max()
        return max_drawdown

    @staticmethod
    def calculate_win_rate(pnl):
        """
        计算胜率。

        :param pnl: 盈亏序列。
        :return: 胜率。
        """
        win_rate = (pnl > 0).mean()
        return win_rate


class Experiment:
    def __init__(self, strategy_class, backtest_class, metrics_class):
        self.strategy_class = strategy_class
        self.backtest_class = backtest_class
        self.metrics_class = metrics_class

    def grid_search(self, predictions, etf_data, param_grid, selection_metric='sharpe_ratio', start_date=None,
                    end_date=None, accumulation=True, intensity_deduct=False):
        """
        网格搜索最优参数。

        :param predictions: 预测值数据。
        :param etf_data: ETF 数据。
        :param param_grid: 参数网格（字典）。
        :param selection_metric: 选择最优策略的指标（如 'sharpe_ratio' 或 'pnl'）。
        :param start_date: 回测起始日期（字符串格式，如 '2020-01-01'）。
        :param end_date: 回测结束日期（字符串格式，如 '2020-12-31'）。
        :return: 最优参数组合及其对应的指标。
        """
        best_params = None
        best_metrics = None
        best_value = -np.inf
        results_table = []
        all_results = {}  # 存储所有策略的结果

        for buy_threshold in param_grid['buy_threshold']:
            for buy_time in param_grid['buy_time']:
                for sell_time in param_grid['sell_time']:
                    strategy = self.strategy_class(predictions, etf_data)
                    strategy.set_parameters(buy_threshold, buy_time, sell_time)

                    backtest = self.backtest_class(
                        strategy=strategy,
                        initial_capital=1e6,
                        transaction_cost=0.001,
                        start_date=start_date,
                        end_date=end_date,
                        accumulation=accumulation,
                        intensity_deduct=intensity_deduct
                    )
                    backtest.run_backtest()
                    results = backtest.get_results()

                    returns = results['pnl'] / results['capital'].shift(1)
                    sharpe_ratio = self.metrics_class.calculate_sharpe_ratio(returns)
                    max_drawdown = self.metrics_class.calculate_max_drawdown(results['capital'])
                    win_rate = self.metrics_class.calculate_win_rate(results['pnl'])
                    total_pnl = results['pnl'].sum()

                    # 计算换手率
                    turnover = len(results['pnl']) / len(predictions)

                    # 记录每组实验结果
                    params_key = f"BT{buy_threshold}_BT{buy_time}_ST{sell_time}"
                    all_results[params_key] = results

                    results_table.append({
                        'buy_threshold': buy_threshold,
                        'buy_time': buy_time,
                        'sell_time': sell_time,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'winrate': win_rate,
                        'total_pnl': total_pnl,
                        'turnover': turnover
                    })

                    if selection_metric == 'sharpe_ratio':
                        metric_value = sharpe_ratio
                    elif selection_metric == 'pnl':
                        metric_value = total_pnl
                    else:
                        raise ValueError(f"未知的选择指标: {selection_metric}")

                    if metric_value > best_value:
                        best_value = metric_value
                        best_params = {'buy_threshold': buy_threshold, 'buy_time': buy_time, 'sell_time': sell_time}
                        best_metrics = {'sharpe_ratio': sharpe_ratio, 'total_pnl': total_pnl}

        # 选择最优策略并可视化
        # best_strategy = self.strategy_class(predictions, etf_data)
        # best_strategy.set_parameters(**best_params)
        # best_backtest = self.backtest_class(best_strategy, start_date=start_date, end_date=end_date)
        # best_backtest.run_backtest()
        # best_results = best_backtest.get_results()
        # self.visualize_results(best_results)

        return pd.DataFrame(results_table), best_params, best_metrics, all_results

    def visualize_results(self, results):
        """
        可视化回测结果。

        :param results: 回测结果 DataFrame。
        """
        # 确保 buy_date 是日期格式
        results['buy_date'] = pd.to_datetime(results['buy_date'], format='%Y%m%d')

        plt.figure(figsize=(12, 8))

        # 子图1：资金曲线
        plt.subplot(2, 1, 1)
        plt.plot(results['buy_date'], results['capital'], label='Capital', color='blue')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.legend()

        # 子图2：每日盈亏柱状图
        plt.subplot(2, 1, 2)
        plt.bar(results['buy_date'], results['pnl'], color='green', alpha=0.6, label='Daily PnL')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.title('Daily Profit and Loss (PnL)')
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 子图3：PnL分布图
        plt.figure(figsize=(12, 6))
        plt.hist(results['pnl'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Daily PnL')

        plt.xlabel('PnL')
        plt.ylabel('Frequency')

        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def compare_strategies(self, all_results):
        """
        对比多个策略的资金曲线。

        :param all_results: 包含多个策略结果的字典。
        """
        plt.figure(figsize=(14, 7))

        for name, results in all_results.items():
            # 确保 buy_date 是日期格式，并去掉小数部分
            try:
                results['buy_date'] = pd.to_datetime(results['buy_date'].astype(str).str.split('.').str[0], format='%Y%m%d')
            except:
                results['buy_date'] = pd.to_datetime(results['buy_date'].astype(str).str.split('.').str[0],
                                                     format='%Y-%m-%d')

            # 确保 capital 是一维数组
            buy_dates = results['buy_date'].values  # 转换为 NumPy 数组
            capital = results['capital'].values  # 转换为 NumPy 数组

            plt.plot(buy_dates, capital, label=name)

        plt.title('Comparison of Portfolio Values Across Strategies')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_best_strategy(self, best_results):
        """
        可视化最优策略的结果。

        :param best_results: 最优策略的回测结果 DataFrame。
        """
        # 确保 buy_date 是日期格式，并去掉小数部分
        best_results['buy_date'] = pd.to_datetime(best_results['buy_date'].astype(str).str.split('.').str[0],
                                                  format='%Y%m%d')

        plt.figure(figsize=(14, 8))

        # 子图1：资金曲线
        plt.subplot(2, 1, 1)
        buy_dates = best_results['buy_date'].values  # 转换为 NumPy 数组
        capital = best_results['capital'].values - 1e6  # 转换为 NumPy 数组
        plt.plot(buy_dates, capital, label='Capital', color='blue')
        plt.title('Best Strategy Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Profit')
        plt.legend()

        # 子图2：每日盈亏柱状图
        plt.subplot(2, 1, 2)
        pnl = best_results['pnl'].values  # 转换为 NumPy 数组
        plt.bar(buy_dates, pnl, color='green', alpha=0.6, label='Daily PnL')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.title('Best Strategy Daily Profit and Loss (PnL)')
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.legend()

        plt.tight_layout()
        plt.show()