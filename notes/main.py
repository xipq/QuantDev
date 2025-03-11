'''
Codebase by qwen
'''


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm


# 新增：分钟级数据获取函数
def get_minute_data(ticker, start_date, end_date, interval='15m'):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    df.index = df.index.tz_convert('Asia/Shanghai')  # 转换时区
    df.index = df.index.normalize()  # 去除非日期部分
    return df


# 读取预测数据
def load_prediction_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['date', 'prediction'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    return df


# 获取中证1000ETF数据（示例代码使用512100.SS）
def get_etf_data(start_date, end_date, ticker='512100.SS'):
    etf = yf.download(ticker, start=start_date, end=end_date)
    etf.index = etf.index.tz_localize(None)  # 去除时区信息
    return etf


# 合并预测数据和ETF价格数据
def merge_data(prediction_df, etf_df):
    merged = pd.concat([prediction_df, etf_df], axis=1, join='inner')
    merged = merged.ffill().dropna()
    return merged


# 计算交易指标
def calculate_metrics(returns, risk_free_rate=0.02):
    cumulative_return = (returns + 1).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
    max_drawdown = (returns + 1).cumprod().div((returns + 1).cumprod().cummax()) - 1
    max_drawdown = max_drawdown.min()

    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


# 回测主函数
def backtest_strategy(merged_data, buy_time='close', sell_time='open_1h',
                      commission=0.0002, stamp_duty=0.001, compound=True):
    """
    参数说明：
    buy_time: 'close'（当日收盘）或 'open'（次日开盘）
    sell_time: 'open_1h'（次日开盘1小时后，用最高价模拟）或 'close'（次日收盘）
    """

    # 生成交易信号
    signals = merged_data['prediction'].shift(1) > 200  # T日14:00生成的预测用于T+1日交易

    # 创建交易记录
    trades = []
    position = 0
    capital = 1000000  # 初始资金
    capital_history = [capital]

    for i in range(1, len(merged_data)):
        current_date = merged_data.index[i]
        prev_date = merged_data.index[i - 1]

        # 获取买入价格
        if buy_time == 'close':
            buy_price = merged_data.loc[prev_date, 'Close']
        elif buy_time == 'open':
            buy_price = merged_data.loc[current_date, 'Open']
        else:
            raise ValueError("Invalid buy_time parameter")

        # 获取卖出价格
        if sell_time == 'open_1h':
            sell_price = merged_data.loc[current_date, 'High']  # 用最高价模拟开盘1小时后的价格
        elif sell_time == 'close':
            sell_price = merged_data.loc[current_date, 'Close']
        else:
            raise ValueError("Invalid sell_time parameter")

        # 计算交易量
        if signals[i]:
            shares = capital // buy_price if position == 0 else 0
        else:
            shares = 0

        # 执行买入
        if shares > 0:
            buy_cost = shares * buy_price * (1 + commission)
            capital -= buy_cost
            position += shares

        # 执行卖出（T+1制度）
        if position > 0:
            sell_amount = position * sell_price
            sell_cost = sell_amount * (1 - commission - stamp_duty)
            capital += sell_cost
            position = 0

        # 记录资金变化
        if compound:
            capital_history.append(capital)
        else:
            capital_history.append(1000000 + (capital - 1000000))

    # 计算收益率
    returns = pd.Series(capital_history).pct_change().dropna()

    # 计算指标
    metrics = calculate_metrics(returns)
    metrics['final_capital'] = capital
    metrics['buy_time'] = buy_time
    metrics['sell_time'] = sell_time

    return metrics, capital_history


# 参数扫描
def parameter_scan(merged_data):
    results = []
    for buy in ['close', 'open']:
        for sell in ['open_1h', 'close']:
            print(f"Testing: Buy {buy}, Sell {sell}")
            metrics, _ = backtest_strategy(merged_data, buy, sell)
            results.append(metrics)

    return pd.DataFrame(results)


# 改进的回测主函数
def backtest_strategy_enhanced(merged_data, minute_data=None,
                               buy_time='close', sell_time='open_1h',
                               commission=0.0002, stamp_duty=0.001,
                               slippage=0.001,  # 新增滑点参数（0.1%）
                               risk_control=False,  # 是否启用动态仓位
                               stop_loss=0.03,  # 新增止损参数（3%）
                               take_profit=0.05,  # 新增止盈参数（5%）
                               compound=True):
    # 生成交易信号
    signals = merged_data['prediction'].shift(1) > 200

    # 创建交易记录
    trades = []
    position = 0
    capital = 1000000  # 初始资金
    capital_history = [capital]

    for i in range(1, len(merged_data)):
        current_date = merged_data.index[i]
        prev_date = merged_data.index[i - 1]

        # 新增：计算波动率（20日收益率标准差）
        volatility = merged_data['Close'].pct_change().rolling(20).std().iloc[i]

        # 动态仓位管理（基于波动率）
        if risk_control:
            position_size = min(0.1 / volatility, 1.0)  # 波动率倒数作为仓位比例
        else:
            position_size = 1.0

        # 新增：滑点处理
        def apply_slippage(price, direction):
            if direction == 'buy':
                return price * (1 + slippage)
            else:
                return price * (1 - slippage)

        # 获取买入价格
        if buy_time == 'close':
            buy_price = merged_data.loc[prev_date, 'Close']
        elif buy_time == 'open':
            buy_price = merged_data.loc[current_date, 'Open']
        elif buy_time == 'vwap':  # 新增：基于分钟数据的VWAP
            if minute_data is None:
                raise ValueError("Minute data required for VWAP")
            vwap = (minute_data.loc[current_date, 'Close'] * minute_data.loc[current_date, 'Volume']).sum() / \
                   minute_data.loc[current_date, 'Volume'].sum()
            buy_price = vwap

        # 应用滑点
        buy_price = apply_slippage(buy_price, 'buy')

        # 获取卖出价格
        if sell_time == 'open_1h':
            if minute_data is not None:
                # 获取开盘后1小时的分钟数据
                morning_data = minute_data.between_time('09:30', '10:30').loc[current_date]
                if not morning_data.empty:
                    sell_price = morning_data['Close'].iloc[-1]  # 使用1小时后的收盘价
                else:
                    sell_price = merged_data.loc[current_date, 'High']  # 备用方案
            else:
                sell_price = merged_data.loc[current_date, 'High']
        elif sell_time == 'close':
            sell_price = merged_data.loc[current_date, 'Close']

        # 应用滑点
        sell_price = apply_slippage(sell_price, 'sell')

        # 止损止盈逻辑
        if position > 0:
            # 计算当前持仓成本
            cost_price = (capital_history[-1] - (capital - position * buy_price)) / position
            # 计算当前价格相对于成本价的变动
            price_change = (sell_price - cost_price) / cost_price
            # 触发止损
            if price_change < -stop_loss:
                sell_price = cost_price * (1 - stop_loss)
            # 触发止盈
            elif price_change > take_profit:
                sell_price = cost_price * (1 + take_profit)

        # 计算交易量
        if signals[i]:
            max_shares = int((capital * position_size) // buy_price)
            shares = max_shares if position == 0 else 0
        else:
            shares = 0

        # 执行买入
        if shares > 0:
            buy_cost = shares * buy_price * (1 + commission)
            capital -= buy_cost
            position += shares

        # 执行卖出（T+1制度）
        if position > 0:
            sell_amount = position * sell_price
            sell_cost = sell_amount * (1 - commission - stamp_duty)
            capital += sell_cost
            position = 0

        # 记录资金变化
        if compound:
            capital_history.append(capital)
        else:
            capital_history.append(1000000 + (capital - 1000000))

    # 计算收益率
    returns = pd.Series(capital_history).pct_change().dropna()

    # 计算指标
    metrics = calculate_metrics(returns)
    metrics['final_capital'] = capital
    metrics['buy_time'] = buy_time
    metrics['sell_time'] = sell_time
    metrics['slippage'] = slippage
    metrics['stop_loss'] = stop_loss
    metrics['take_profit'] = take_profit

    return metrics, capital_history


# 新增：样本外测试函数
def walk_forward_test(prediction_df, etf_df, minute_df=None, windows=3):
    results = []
    total_length = len(prediction_df)
    window_size = total_length // windows

    for i in range(windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i != windows - 1 else len(prediction_df)

        # 划分样本内外数据
        in_sample_pred = prediction_df.iloc[:end_idx]
        out_sample_pred = prediction_df.iloc[end_idx:]

        # 合并数据
        in_sample = merge_data(in_sample_pred, etf_df)
        out_sample = merge_data(out_sample_pred, etf_df)

        # 在样本内数据优化参数
        best_params = optimize_parameters(in_sample)  # 需要实现优化函数

        # 在样本外数据测试
        metrics, _ = backtest_strategy_enhanced(out_sample, minute_df, **best_params)
        metrics['window'] = i + 1
        results.append(metrics)

    return pd.DataFrame(results)


# 新增：过拟合检测函数
def overfitting_detection(returns):
    # 计算过拟合概率（简化版）
    sharpe = returns.mean() / returns.std()
    autocorr = returns.autocorr()
    pbo = norm.cdf(-(sharpe * np.sqrt(252) + autocorr))

    return {
        'prob_overfitting': pbo,
        'autocorrelation': autocorr
    }


# 主程序
if __name__ == "__main__":
    # 加载数据
    prediction_df = load_prediction_data('../data/score.csv')
    start_date = prediction_df.index.min().strftime('%Y-%m-%d')
    end_date = (prediction_df.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')

    # 获取ETF日线数据
    etf_df = get_etf_data(start_date, end_date, ticker='512100.SS')

    # 新增：获取分钟级数据（需要网络连接）
    minute_df = get_minute_data('512100.SS', start_date, end_date, interval='15m')

    # 合并数据
    merged_data = merge_data(prediction_df, etf_df)

    # 执行增强回测
    metrics, capital = backtest_strategy_enhanced(
        merged_data,
        minute_data=minute_df,
        buy_time='open',
        sell_time='open_1h',
        slippage=0.0015,
        risk_control=True,
        stop_loss=0.02,
        take_profit=0.04
    )

    # 执行样本外测试
    #wf_results = walk_forward_test(prediction_df, etf_df, minute_df)

    # 过拟合检测
    returns = pd.Series(capital).pct_change().dropna()
    of_metrics = overfitting_detection(returns)

    # 输出结果
    print("Enhanced Metrics:", metrics)
    #print("Walk Forward Results:\n", wf_results)
    print("Overfitting Metrics:", of_metrics)

    # 绘制资金曲线
    plt.figure(figsize=(14, 7))
    plt.plot(capital, label='Enhanced Strategy')
    plt.title('Enhanced Strategy Performance with Risk Control')
    plt.show()