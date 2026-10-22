"""因子计算：输入长表行情，输出每个交易日、每只股票的因子值。

行情字段约定（download.py 产出）：date, symbol, open, close, high, low,
volume, amount, turnover。全部按 date 升序。
"""
import numpy as np
import pandas as pd

import config


def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表转宽表：index=date, columns=symbol。"""
    return df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")


def cross_section_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 z-score 标准化（去极值用 mad 缩尾）。"""
    def _z(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 5:
            return pd.Series(np.nan, index=row.index)
        med = valid.median()
        mad = (valid - med).abs().median()
        if mad == 0:
            return pd.Series(np.nan, index=row.index)
        clipped = valid.clip(med - 3 * 1.4826 * mad, med + 3 * 1.4826 * mad)
        sd = clipped.std(ddof=0)
        if sd == 0:
            return pd.Series(np.nan, index=row.index)
        return (clipped - clipped.mean()) / sd

    return wide.apply(_z, axis=1)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算动量、低波动、低换手三类因子，返回宽表字典。

    停牌日行情为空，先做前向填充再计算，避免把停牌当成暴跌。
    """
    close = pivot(df, "close").ffill()
    turnover = pivot(df, "turnover").ffill()
    amount = pivot(df, "amount").ffill()

    # 1. 动量：过去 N 日涨幅，越大越好
    momentum = close / close.shift(config.MOM_WINDOW) - 1

    # 2. 低波动：过去 N 日收益率标准差，越小越好
    ret = close.pct_change(fill_method=None)
    volatility = ret.rolling(config.VOL_WINDOW, min_periods=int(config.VOL_WINDOW * 0.8)).std()

    # 3. 低换手：过去 N 日平均换手率，越小越好
    avg_turnover = turnover.rolling(config.TURN_WINDOW, min_periods=int(config.TURN_WINDOW * 0.8)).mean()

    # 4. 流动性过滤用的均成交额
    avg_amount = amount.rolling(20, min_periods=15).mean()

    return {
        "momentum": momentum,
        "volatility": volatility,
        "avg_turnover": avg_turnover,
        "avg_amount": avg_amount,
    }


def composite_score(factors: dict) -> pd.DataFrame:
    """把多个因子标准化后按权重合成打分，越大越好。"""
    z_mom = cross_section_zscore(factors["momentum"])
    z_vol = cross_section_zscore(-factors["volatility"])       # 取负 -> 低波动得分高
    z_turn = cross_section_zscore(-factors["avg_turnover"])     # 取负 -> 低换手得分高

    w = config.FACTOR_WEIGHTS
    score = (
        z_mom.fillna(0) * w["momentum"]
        + z_vol.fillna(0) * w["low_vol"]
        + z_turn.fillna(0) * w["low_turnover"]
    )
    # 原始因子全为 NaN 的股票不应入选
    valid = z_mom.notna() | z_vol.notna() | z_turn.notna()
    score = score.where(valid)
    return score
