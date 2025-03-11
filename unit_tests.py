import unittest
import pandas as pd
from Strategy import Strategy
from Backtest import Backtest


class TestBacktestTimeRange(unittest.TestCase):
    def setUp(self):
        # 模拟预测值和ETF数据
        self.predictions = pd.DataFrame({
            'date': ['20200102', '20200103', '20200106', '20200107'],
            'prediction': [300, 400, 500, 600]
        })
        self.etf_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-02', periods=10, freq='D'),
            'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        }).set_index('date')

    def test_time_range_filtering(self):
        # 初始化策略
        strategy = Strategy(self.predictions, self.etf_data)
        strategy.set_parameters(buy_threshold=200, buy_time='14:30', sell_time='09:31')

        # 设置回测时间范围
        backtest = Backtest(strategy, start_date='2020-01-03', end_date='2020-01-06')
        backtest.run_backtest()

        # 获取结果
        results = backtest.get_results()

        # 验证结果是否在指定时间范围内
        self.assertTrue((results['buy_date'] >= '20200103').all())
        self.assertTrue((results['sell_date'] <= '20200106').all())


if __name__ == "__main__":
    unittest.main()