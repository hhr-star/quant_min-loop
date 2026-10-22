"""绩效指标：全部基于日频净值序列计算。"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(nav: pd.Series) -> pd.Series:
    """净值序列 -> 日收益率序列。"""
    return nav.pct_change(fill_method=None).dropna()


def annualized_return(nav: pd.Series) -> float:
    """年化收益率（几何）。"""
    if len(nav) < 2:
        return np.nan
    total = nav.iloc[-1] / nav.iloc[0]
    years = len(nav) / TRADING_DAYS
    return total ** (1 / years) - 1


def annualized_vol(nav: pd.Series) -> float:
    """年化波动率。"""
    rets = to_returns(nav)
    return rets.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe(nav: pd.Series, rf: float = 0.0) -> float:
    """夏普比率，rf 为年化无风险利率。"""
    rets = to_returns(nav)
    if rets.std(ddof=1) == 0:
        return np.nan
    excess = rets - rf / TRADING_DAYS
    return excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（负值）。"""
    peak = nav.cummax()
    dd = nav / peak - 1
    return dd.min()


def calmar(nav: pd.Series) -> float:
    """Calmar = 年化收益 / |最大回撤|。"""
    mdd = max_drawdown(nav)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(nav) / abs(mdd)


def win_rate(nav: pd.Series) -> float:
    """日胜率。"""
    rets = to_returns(nav)
    return (rets > 0).mean()


def monthly_returns(nav: pd.Series) -> pd.Series:
    """月度收益率。"""
    m = nav.resample("ME").last()
    return m.pct_change(fill_method=None).dropna()


def yearly_returns(nav: pd.Series) -> pd.Series:
    """年度收益率。"""
    y = nav.resample("YE").last()
    first = nav.iloc[0]
    out = {}
    prev = first
    for ts, val in y.items():
        out[ts.year] = val / prev - 1
        prev = val
    return pd.Series(out)


def summarize(nav: pd.Series, turnover: pd.Series | None = None) -> pd.Series:
    """汇总核心指标。"""
    data = {
        "累计收益": nav.iloc[-1] / nav.iloc[0] - 1,
        "年化收益": annualized_return(nav),
        "年化波动": annualized_vol(nav),
        "夏普比率": sharpe(nav),
        "最大回撤": max_drawdown(nav),
        "Calmar": calmar(nav),
        "日胜率": win_rate(nav),
        "月胜率": (monthly_returns(nav) > 0).mean(),
    }
    if turnover is not None and len(turnover) > 0:
        data["年均单边换手"] = turnover.mean() * TRADING_DAYS / 2
    return pd.Series(data)
