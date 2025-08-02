# utils.py

import numpy as np
import pandas as pd


def get_intraday_range(high, low):
    """日内振幅 = 最高价 - 最低价"""
    return high - low


def get_normalized_close_position(close, high, low):
    """归一化收盘位置 = (收盘价 - 最低价) / (最高价 - 最低价)"""
    return (close - low) / (high - low + 1e-6)


def get_intraday_risk_indicator(high, low):
    """日内风险指标 = (最高价 - 最低价)^2"""
    return np.square(high - low)


def get_closing_strength(close, open_, high, low):
    """收盘强弱 = abs(收盘价 - 开盘价) / (最高价 - 最低价)"""
    return np.abs(close - open_) / (high - low + 1e-6)


def get_log_range_with_body_strength(close, open_, high, low):
    """
    对数振幅 x 实体波动
    = ln(最高价 / 最低价) * abs(收盘 - 开盘)
    """
    log_range = np.log(high / (low + 1e-6))
    body_strength = np.abs(close - open_)
    return log_range * body_strength


def get_upper_shadow(high, open_, close):
    """上影线长度 = High - max(Open, Close)"""
    return high - np.maximum(open_, close)


def get_lower_shadow(low, open_, close):
    """下影线长度 = min(Open, Close) - Low"""
    return np.minimum(open_, close) - low


def get_body_length(open_, close):
    """实体长度 = |Close - Open|"""
    return np.abs(close - open_)


def get_shadow_ratio(high, low, open_, close):
    """
    影线比例 = (上影线 + 下影线) / (实体 + 1e-6)
    用于衡量多空争夺强度（非线性组合）
    """
    upper = get_upper_shadow(high, open_, close)
    lower = get_lower_shadow(low, open_, close)
    body = get_body_length(open_, close)
    return (upper + lower) / (body + 1e-6)


def get_stochastic_k(close, high_n, low_n):
    """
    随机指标 K 值 (%K)
    %K = (Close - Low_n) / (High_n - Low_n) * 100
    通常 n = 9
    """
    return (close - low_n) / (high_n - low_n + 1e-6) * 100


def get_stochastic_d(k_values, m=3):
    """
    随机指标 D 值 (%D)
    %D = SMA(%K, m)，常用 m=3 平滑天数
    """
    return k_values.rolling(window=m).mean()


# 用于批量计算 %K 和 %D
def get_kdj(df, n=9, m=3):
    """
    计算 KDJ 指标(%K 和 %D)
    - n: 计算周期
    - m: 平滑天数

    返回: df[['%K', '%D']]
    """
    low_n = df["Low"].rolling(window=n).min()
    high_n = df["High"].rolling(window=n).max()

    k = get_stochastic_k(df["Close"], high_n, low_n)
    d = get_stochastic_d(k, m)

    return pd.DataFrame({"%K": k, "%D": d})
