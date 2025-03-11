import pandas as pd
from Strategy import Strategy
from Backtest import Backtest
from Utils import Experiment, Metrics


if __name__ == "__main__":
    # 加载数据
    predictions = pd.read_csv('data/score.csv', names=['date', 'prediction'])
    etf_data = pd.read_csv('data/1000ETF1m_new.csv', parse_dates=['date'], index_col='date')

    # param_grid = {
    #     'buy_threshold': [200,300,400,500],
    #     'buy_time': ['14:30', '15:00'],
    #     'sell_time': ['09:31', '10:30', '14:00']
    # }
    #

    param_grid = {
        'buy_threshold': [200,500],
        'buy_time': ['same_day_14:30', 'same_day_14:50'],
        'sell_time': ['next_day_10:00', 'next_day_10:30', 'next_day_14:00', 'next_day_14:30']
    }

    experiment = Experiment(Strategy, Backtest, Metrics)

    results_table, best_params, best_metrics, all_results = experiment.grid_search(
        predictions, etf_data, param_grid,
        selection_metric='sharpe_ratio',
        start_date='2020-01-01',
        end_date='2024-12-31',
        accumulation=True
    )

    print("实验结果表：")
    print(results_table)
    # save to csv
    results_table.to_csv('results_table_single.csv', index=False)

    # 根据sharpe，max_drawdown，win_rate等指标选择最优策略
    print("Best Sharpe Strategy:", best_params, best_metrics)

    # min dd and max winrate, sort and retrieve from results_table
    best_sharpe_strategy = results_table.sort_values(by=['sharpe_ratio', 'max_drawdown', 'winrate'], ascending=[False, True, False]).iloc[0]
    print("Best Sharpe Strategy:", best_sharpe_strategy)
    best_sharpe_strategy.to_csv('best_sharpe_strategy_single.csv', index=False)

    best_dd_strategy = results_table.sort_values(by=['max_drawdown', 'sharpe_ratio', 'winrate'], ascending=[True, False, False]).iloc[0]
    print("Best DD Strategy:", best_dd_strategy)
    best_dd_strategy.to_csv('best_dd_strategy_single.csv', index=False)

    best_wr_strategy = results_table.sort_values(by=['winrate', 'sharpe_ratio', 'max_drawdown'], ascending=[False, False, True]).iloc[0]
    print("Best WR Strategy:", best_wr_strategy)
    best_wr_strategy.to_csv('best_wr_strategy_single.csv', index=False)


    # 获取最优策略的结果
    best_strategy_key = f"BT{best_params['buy_threshold']}_BT{best_params['buy_time']}_ST{best_params['sell_time']}"
    best_results = all_results[best_strategy_key]

    # 可视化最优策略
    experiment.visualize_best_strategy(best_results)

    # 对比所有策略的资金曲线
    experiment.compare_strategies(all_results)