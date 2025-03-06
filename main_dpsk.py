'''
Codebase by deepseek
'''
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 配置参数
INIT_CAPITAL = 1e6  # 初始资金
FEE_RATE = 0.002  # 单边交易费率(含佣金税费)
TRADE_RATIO = 1.0  # 仓位比例
COMPOUND = True  # 是否复利

# 策略参数扫描空间
BUY_TIMES = ['close']  # 买入时机选择
SELL_TIMES = ['open', 'open_1h', 'close']  # 卖出时机选择

# 加载预测数据
pred_df = pd.read_csv('score.csv', header=None, names=['date', 'score'])
pred_df['date'] = pd.to_datetime(pred_df['date'], format='%Y%m%d')

# 获取中证1000指数数据
index_df = ak.stock_zh_index_hist_csindex(symbol="H00922")
index_df['date'] = pd.to_datetime(index_df['日期'])
index_df = index_df[['date', '开盘', '最高', '最低', '收盘']]
index_df.columns = ['date', 'open', 'high', 'low', 'close']

# 合并数据
merged = pd.merge_asof(pred_df.sort_values('date'),
                       index_df.sort_values('date'),
                       on='date')

# 生成交易信号
merged['signal'] = np.where(merged['score'] > 200, 1, 0)


def backtest_strategy(df, buy_time, sell_time):
    capital = INIT_CAPITAL
    position = 0
    returns = []
    trade_log = []

    for i in range(len(df) - 1):
        current_date = df.iloc[i]['date']
        next_date = df.iloc[i + 1]['date']

        if df.iloc[i]['signal'] == 1:
            # 获取买入价格
            if buy_time == 'close':
                buy_price = df.iloc[i]['close']
            else:
                buy_price = df.iloc[i]['open']

            # 计算可买数量
            if COMPOUND:
                position = capital * TRADE_RATIO / buy_price
            else:
                position = INIT_CAPITAL * TRADE_RATIO / buy_price

            # 记录交易
            trade_log.append({
                'date': current_date,
                'type': 'buy',
                'price': buy_price,
                'shares': position
            })

            # 获取卖出价格
            if sell_time == 'open':
                sell_price = df.iloc[i + 1]['open']
            elif sell_time == 'open_1h':
                # 假设开盘1小时价格为当日最高最低的平均值（需实际数据需调整）
                sell_price = (df.iloc[i + 1]['high'] + df.iloc[i + 1]['low']) / 2
            else:
                sell_price = df.iloc[i + 1]['close']

            # 计算收益
            pct = (sell_price * (1 - FEE_RATE)) / (buy_price * (1 + FEE_RATE)) - 1
            trade_return = position * buy_price * pct

            # 更新资金
            if COMPOUND:
                capital += trade_return
            else:
                capital = INIT_CAPITAL + trade_return

            # 记录收益
            returns.append({
                'date': next_date,
                'return': trade_return,
                'buy_price': buy_price,
                'sell_price': sell_price
            })

    # 计算绩效指标
    if returns:
        ret_df = pd.DataFrame(returns)
        annual_ret = ret_df['return'].sum() / INIT_CAPITAL * 252 / len(ret_df)
        vol = ret_df['return'].std() * np.sqrt(252)
        sharpe = annual_ret / vol if vol != 0 else 0
    else:
        annual_ret = vol = sharpe = 0

    return {
        'buy_time': buy_time,
        'sell_time': sell_time,
        'annual_return': annual_ret,
        'volatility': vol,
        'sharpe': sharpe,
        'trades': len(returns),
        'final_value': capital
    }


# 执行参数扫描
results = []
for buy_time in BUY_TIMES:
    for sell_time in SELL_TIMES:
        res = backtest_strategy(merged, buy_time, sell_time)
        results.append(res)

# 展示结果
result_df = pd.DataFrame(results)
print("策略回测结果：")
print(result_df[['buy_time', 'sell_time', 'annual_return', 'volatility', 'sharpe', 'trades']])

# 可视化资金曲线
plt.figure(figsize=(12, 6))
for strategy in results:
    label = f"{strategy['buy_time']}-{strategy['sell_time']}"
    plt.plot([INIT_CAPITAL, strategy['final_value']], label=label)
plt.legend()
plt.title('Strategy Performance Comparison')
plt.ylabel('Portfolio Value')
plt.show()