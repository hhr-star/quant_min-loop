"""全局参数配置：所有可调参数集中在这里，便于做参数敏感性分析。"""

# ---------- 回测区间 ----------
START_DATE = "2019-01-01"
END_DATE = "2026-10-21"

# 样本内 / 样本外切分点（此后为样本外，不参与任何参数选择）
IN_SAMPLE_END = "2022-12-31"

# ---------- 股票池 ----------
INDEX_CODE = "000300"          # 沪深300；可改 000905（中证500）、000852（中证1000）
MIN_LISTED_DAYS = 250          # 上市满 250 个交易日才纳入，规避新股
EXCLUDE_ST = True              # 剔除 ST / *ST
MIN_AMOUNT = 2000              # 过去 20 日均成交额下限（万元），剔除流动性差的票

# ---------- 因子 ----------
MOM_WINDOW = 20                # 动量窗口（交易日）
VOL_WINDOW = 60                # 波动率窗口
TURN_WINDOW = 20               # 换手率窗口
FACTOR_WEIGHTS = {
    "momentum": 1.0,           # 动量：越大越好
    "low_vol": 1.0,            # 低波动：取负号后越大越好
    "low_turnover": 1.0,       # 低换手：取负号后越大越好
}

# ---------- 组合 ----------
TOP_N = 50                     # 持仓数量
REBALANCE = "M"                # 调仓频率：M 月末 / W 每周五
WEIGHTING = "equal"            # equal 等权 / score 按打分加权

# ---------- 交易成本 ----------
COMMISSION = 0.00025           # 佣金，双边各收
STAMP_TAX = 0.0005             # 印花税，仅卖出
SLIPPAGE = 0.001               # 滑点，单边
MIN_COMMISSION = 5.0           # 单笔最低佣金（元），在按金额比例成本外近似忽略

# ---------- 输出 ----------
DATA_DIR = "data"
REPORT_DIR = "reports"
