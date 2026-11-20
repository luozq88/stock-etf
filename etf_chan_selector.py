#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股 ETF：月线多头 + 周线回踩条件 + T+0/T+1 分类 + DeepSeek 分析

选股条件：
1) 月线多头：最新月线收盘价同时大于 5/10/20 月均线（不要求均线多头排列）；
2) 周线条件：最近 3 个交易日中，至少有一天收盘价同时大于 5 周线、10 周线，
   但小于 20 周线；
3) 综合打分后取前 10 只，调用 DeepSeek（flash 版）做结构化分析。

说明：月线量能不再作为硬性条件，仅在结果和 DeepSeek 提示词中作为参考指标。
数据源：Tushare Pro（fund_basic / fund_daily / fund_adj / trade_cal）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import tushare as ts

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODELS = [
    os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    "deepseek-v4-pro",
]

MIN_MONTHS = 21              # 月线至少 21 根，才能算 20 月均线
MIN_AVG_AMOUNT_YI = 0.02     # 近 20 日日均成交额下限（亿元）
WEEKLY_CHECK_DAYS = 3        # 最近 N 个交易日中，至少一天满足周线条件


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 二、Tushare API 封装
# ----------------------------------------------------------------------------
class Api:
    def __init__(self, token: str):
        if not token:
            raise RuntimeError("请先设置 TUSHARE_TOKEN 环境变量")
        ts.set_token(token)
        self.pro = ts.pro_api()

    def call(self, fn, **kwargs):
        last_err = None
        for attempt in range(6):
            try:
                return fn(**kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                text = str(e)
                if any(k in text for k in ["每分钟", "每小时", "频率", "limit", "too many"]):
                    time.sleep(15)
                else:
                    time.sleep(2)
                if attempt >= 2:
                    log(f"  API 重试 {attempt + 1}/6：{text[:120]}")
        log(f"  API 最终失败：{str(last_err)[:200]}")
        return None


# ----------------------------------------------------------------------------
# 三、ETF 股票池 + T+0/T+1 分类
# ----------------------------------------------------------------------------
def classify_trade_type(name: str, fund_type: str, invest_type: str) -> tuple[str, str]:
    name = str(name or "")
    fund_type = str(fund_type or "")
    invest_type = str(invest_type or "")
    text = f"{name} {fund_type} {invest_type}"

    if any(k in fund_type for k in ["货币", "债券"]) or "QDII" in text:
        return "T+0", "债券/货币/QDII 类 ETF，支持 T+0 回转交易"
    if "黄金股" in name:
        return "T+1", "黄金股 ETF，投资 A 股股票，T+1"

    t0_kw = [
        "纳斯达克", "标普", "恒生", "港股", "恒指", "日经", "德国", "法国",
        "美国", "中概", "海外", "QDII", "国债", "政金债", "信用债", "短融",
        "城投债", "可转债", "黄金", "白银", "豆粕", "原油", "期货", "商品",
        "货币",
    ]
    if any(k in name for k in t0_kw):
        return "T+0", "跨境/债券/黄金商品/货币类 ETF，支持 T+0 回转交易"
    return "T+1", "A 股股票型 ETF，T+1 交易"


def load_etf_universe(api: Api) -> pd.DataFrame:
    df = api.call(
        api.pro.fund_basic,
        market="E",
        fields="ts_code,name,fund_type,invest_type,list_date,delist_date",
    )
    if df is None or df.empty:
        raise RuntimeError("fund_basic 返回为空")

    df = df.copy()
    if "delist_date" in df.columns:
        df = df[df["delist_date"].isna() | (df["delist_date"].astype(str) == "")]

    df["name"] = df["name"].astype(str)
    for c in ["fund_type", "invest_type"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)

    classified = df.apply(
        lambda r: classify_trade_type(r["name"], r["fund_type"], r["invest_type"]),
        axis=1,
    )
    df["trade_type"] = [x[0] for x in classified]
    df["trade_type_reason"] = [x[1] for x in classified]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 四、交易日历与月线预筛
# ----------------------------------------------------------------------------
def month_end_trade_dates(api: Api, start_date: str, end_date: str) -> list[str]:
    cal = api.call(
        api.pro.trade_cal,
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="cal_date",
    )
    if cal is None or cal.empty:
        return []
    days = sorted(cal["cal_date"].astype(str).tolist())
    out: dict[str, str] = {}
    for d in days:
        out[d[:6]] = d
    return sorted(out.values())


def detect_latest_trade_date(api: Api, lookback_days: int = 400) -> str:
    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = api.call(
        api.pro.trade_cal,
        exchange="SSE",
        start_date=start,
        end_date=today,
        is_open="1",
        fields="cal_date",
    )
    if cal is None or cal.empty:
        raise RuntimeError("trade_cal 返回为空")
    days = sorted(cal["cal_date"].astype(str), reverse=True)
    for d in days:
        df = api.call(api.pro.fund_daily, trade_date=d, fields="ts_code,close")
        if df is not None and len(df) > 100:
            return d
    raise RuntimeError("未找到有 ETF 行情的最近交易日")


def prefilter_by_monthly_ma(api: Api, universe: pd.DataFrame, latest: str,
                            lookback_months: int = 36) -> pd.DataFrame:
    start_dt = datetime.strptime(latest, "%Y%m%d") - timedelta(days=lookback_months * 31 + 40)
    month_ends = month_end_trade_dates(api, start_dt.strftime("%Y%m%d"), latest)[-lookback_months:]
    log(f"预筛：拉取 {len(month_ends)} 个月末交易日 ETF 收盘价 ...")

    frames = []
    last_snapshot = None
    for i, d in enumerate(month_ends, 1):
        df = api.call(
            api.pro.fund_daily,
            trade_date=d,
            fields="ts_code,close,amount,vol",
        )
        if df is None or df.empty:
            continue
        df = df.copy()
        df["month"] = d[:6]
        frames.append(df[["ts_code", "month", "close"]])
        last_snapshot = df[["ts_code", "close", "amount", "vol"]]
        if i % 10 == 0 or i == len(month_ends):
            log(f"  月末快照 {i}/{len(month_ends)}")

    if not frames:
        raise RuntimeError("月末 ETF 行情为空")

    px = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["ts_code", "month"], keep="last")
        .pivot_table(index="ts_code", columns="month", values="close", aggfunc="last")
        .sort_index(axis=1)
    )
    log(f"  收盘价矩阵：{px.shape[0]} 只 ETF × {px.shape[1]} 个月")

    keep = []
    for code, row in px.iterrows():
        arr = row.dropna().values.astype(float)
        if len(arr) < 20:
            continue
        last = arr[-1]
        if not np.isfinite(last) or last <= 0:
            continue
        if last > arr[-5:].mean() and last > arr[-10:].mean() and last > arr[-20:].mean():
            keep.append(code)

    log(f"  月线均线预筛通过：{len(keep)} 只")
    res = universe[universe["ts_code"].isin(keep)].copy()
    if last_snapshot is not None:
        res = res.merge(
            last_snapshot.rename(columns={
                "close": "latest_close",
                "amount": "latest_amount",
                "vol": "latest_vol",
            }),
            on="ts_code",
            how="left",
        )
    return res


# ----------------------------------------------------------------------------
# 五、日线（前复权）与月线/周线
# ----------------------------------------------------------------------------
def fetch_daily(api: Api, ts_code: str, start_date: str, end_date: str,
                use_adj: bool = True) -> pd.DataFrame | None:
    df = api.call(
        api.pro.fund_daily, ts_code=ts_code, start_date=start_date, end_date=end_date
    )
    if df is None or len(df) < 80:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str)

    if use_adj:
        try:
            adj = api.call(
                api.pro.fund_adj,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="trade_date,adj_factor",
            )
            if adj is not None and len(adj):
                adj = adj.drop_duplicates("trade_date")
                adj["trade_date"] = adj["trade_date"].astype(str)
                df = df.merge(adj, on="trade_date", how="left")
                df["adj_factor"] = (
                    df["adj_factor"].ffill().bfill().fillna(1.0).astype(float)
                )
                for c in ["open", "high", "low", "close", "pre_close"]:
                    if c in df.columns:
                        df[c] = df[c].astype(float) * df["adj_factor"]
        except Exception as e:  # noqa: BLE001
            log(f"    {ts_code} 复权因子获取失败，使用未复权价: {str(e)[:80]}")

    for c in ["open", "high", "low", "close", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(df) < 80:
        return None
    df["pct_chg"] = df["close"].pct_change() * 100.0
    return df


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["dt"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
    d = d.set_index("dt").sort_index()
    m = d.resample("ME").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vol=("vol", "sum"),
        amount=("amount", "sum"),
    )
    return m.dropna(subset=["close"])


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["dt"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
    d = d.set_index("dt").sort_index()
    w = d.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vol=("vol", "sum"),
        amount=("amount", "sum"),
    )
    return w.dropna(subset=["close"])


def weekly_pullback(daily: pd.DataFrame, days: int = WEEKLY_CHECK_DAYS):
    """
    最近 days 个交易日中，是否存在一天：
      收盘价 > 5周线 且 收盘价 > 10周线 且 收盘价 < 20周线。

    对每个交易日，用截至当天的日线重采样周线，计算周线 MA5/10/20，
    得到的周线均线就是当天盘后图表上看到的值。
    """
    n = len(daily)
    if n < 120:
        return None

    hits = []
    for offset in range(days):
        i = n - 1 - offset
        if i < 0:
            break
        w = to_weekly(daily.iloc[:i + 1])
        if len(w) < 20:
            continue
        c = float(daily["close"].iloc[i])
        ma5 = float(w["close"].iloc[-5:].mean())
        ma10 = float(w["close"].iloc[-10:].mean())
        ma20 = float(w["close"].iloc[-20:].mean())
        hits.append({
            "date": str(daily["trade_date"].iloc[i]),
            "close": c,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "passed": bool(c > ma5 and c > ma10 and c < ma20),
        })

    if not hits:
        return None

    hit_records = [h for h in hits if h["passed"]]
    latest = hits[0]
    hit = hit_records[0] if hit_records else latest

    return {
        "passed": bool(hit_records),
        "hit_days": len(hit_records),
        "checked_days": len(hits),
        "hit_date": hit["date"],
        "hit_close": hit["close"],
        "hit_ma5": hit["ma5"],
        "hit_ma10": hit["ma10"],
        "hit_ma20": hit["ma20"],
        "cur_date": latest["date"],
        "cur_close": latest["close"],
        "cur_ma5": latest["ma5"],
        "cur_ma10": latest["ma10"],
        "cur_ma20": latest["ma20"],
        "records": hits,
    }


# ----------------------------------------------------------------------------
# 六、月线波段起点与量能（仅参考）
# ----------------------------------------------------------------------------
def find_rally_start(monthly: pd.DataFrame, lookback: int = 48) -> int | None:
    n = len(monthly)
    if n < 6:
        return None
    start = max(0, n - lookback)
    lows = monthly["low"].iloc[start:].astype(float)
    if lows.isna().all():
        return None
    i = int(np.nanargmin(lows.values)) + start
    if n - 1 - i < 3:
        return None
    latest_close = float(monthly["close"].iloc[-1])
    if latest_close <= float(monthly["low"].iloc[i]):
        return None
    return i


def volume_trend(monthly: pd.DataFrame, rally_i: int, drop_last: bool = False):
    seg = monthly.iloc[rally_i:].copy()
    if drop_last and len(seg) > 1:
        seg = seg.iloc[:-1]
    v = pd.to_numeric(seg["vol"], errors="coerce").dropna().values.astype(float)
    if len(v) < 4 or v.sum() <= 0:
        return None

    x = np.arange(len(v), dtype=float)
    y = np.log1p(v)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    slope_pct = float(np.expm1(slope))
    k = max(1, len(v) // 3)
    early = float(v[:k].mean())
    recent = float(v[-k:].mean())
    ratio = recent / early if early > 0 else 0.0
    med = float(np.median(v))
    max_med = float(v.max() / med) if med > 0 else 999.0

    def clip01(z):
        return float(min(1.0, max(0.0, z)))

    passed = (
        slope_pct >= 0.005
        and r2 >= 0.15
        and ratio >= 1.05
        and max_med <= 5.0
    )
    score = (
        0.50 * clip01(slope_pct / 0.03)
        + 0.25 * clip01(r2)
        + 0.25 * clip01((ratio - 1.0) / 0.5)
    )
    return {
        "slope_pct": slope_pct,
        "r2": r2,
        "recent_early_ratio": ratio,
        "max_median_ratio": max_med,
        "months": len(v),
        "score": score if passed else 0.0,
        "passed": bool(passed),
    }


# ----------------------------------------------------------------------------
# 七、单只 ETF 评估
# ----------------------------------------------------------------------------
def evaluate_etf(api: Api, row: dict, start_date: str, latest: str,
                 use_adj: bool, month_complete: bool):
    code = row["ts_code"]
    daily = fetch_daily(api, code, start_date, latest, use_adj=use_adj)
    if daily is None:
        return None

    monthly = to_monthly(daily)
    if len(monthly) < MIN_MONTHS:
        return None

    close = float(monthly["close"].iloc[-1])
    ma5 = float(monthly["close"].iloc[-5:].mean())
    ma10 = float(monthly["close"].iloc[-10:].mean())
    ma20 = float(monthly["close"].iloc[-20:].mean())
    if not (close > ma5 and close > ma10 and close > ma20):
        return None

    rally_i = find_rally_start(monthly)
    if rally_i is None:
        return None

    # 月线量能仅作参考，不再作为筛选条件
    vt = volume_trend(monthly, rally_i, drop_last=not month_complete)
    if vt is None:
        vt = {
            "slope_pct": np.nan,
            "r2": np.nan,
            "recent_early_ratio": np.nan,
            "max_median_ratio": np.nan,
            "months": 0,
            "score": 0.0,
        }

    wp = weekly_pullback(daily)
    if wp is None or not wp["passed"]:
        return None

    rally_low = float(monthly["low"].iloc[rally_i])
    rally_months = len(monthly) - rally_i
    rally_ret = close / rally_low - 1.0

    dev20 = close / ma20 - 1.0
    dev10 = close / ma10 - 1.0
    dev5 = close / ma5 - 1.0

    def clip01(x):
        return float(min(1.0, max(0.0, x)))

    trend_score = (
        0.60 * clip01(dev20 / 0.20)
        + 0.25 * clip01(dev10 / 0.10)
        + 0.15 * clip01(dev5 / 0.05)
    )

    amt20 = float(daily["amount"].tail(20).mean())
    amt_yi = amt20 / 1e5
    if amt_yi < MIN_AVG_AMOUNT_YI:
        return None
    liq_score = clip01(amt_yi / 2.0)

    total = 70.0 * trend_score + 30.0 * liq_score
    pct = float(daily["pct_chg"].iloc[-1]) if "pct_chg" in daily.columns else np.nan

    return {
        "ts_code": code,
        "name": row["name"],
        "fund_type": row.get("fund_type", ""),
        "invest_type": row.get("invest_type", ""),
        "list_date": row.get("list_date", ""),
        "trade_type": row.get("trade_type", ""),
        "trade_type_reason": row.get("trade_type_reason", ""),
        "close": close,
        "pct_chg": pct,
        "monthly_ma5": ma5,
        "monthly_ma10": ma10,
        "monthly_ma20": ma20,
        "close_vs_ma20": dev20,
        "rally_start": monthly.index[rally_i].strftime("%Y-%m-%d"),
        "rally_months": rally_months,
        "rally_return": rally_ret,
        "vol_slope_pct": vt["slope_pct"],
        "vol_r2": vt["r2"],
        "vol_recent_early": vt["recent_early_ratio"],
        "vol_max_median": vt["max_median_ratio"],
        "vol_months": vt["months"],
        "weekly_hit_days": wp["hit_days"],
        "weekly_checked_days": wp["checked_days"],
        "weekly_hit_date": wp["hit_date"],
        "weekly_hit_close": wp["hit_close"],
        "weekly_hit_ma5": wp["hit_ma5"],
        "weekly_hit_ma10": wp["hit_ma10"],
        "weekly_hit_ma20": wp["hit_ma20"],
        "weekly_cur_date": wp["cur_date"],
        "weekly_cur_close": wp["cur_close"],
        "weekly_cur_ma5": wp["cur_ma5"],
        "weekly_cur_ma10": wp["cur_ma10"],
        "weekly_cur_ma20": wp["cur_ma20"],
        "avg_amount_20d_yi": amt_yi,
        "trend_score": trend_score,
        "liquidity_score": liq_score,
        "score": total,
    }


# ----------------------------------------------------------------------------
# 八、DeepSeek
# ----------------------------------------------------------------------------
def call_deepseek(
    prompt: str,
    system_prompt: str = "你是严谨的 ETF 与缠论技术分析师，只基于给定数据推理，不编造数据。",
) -> tuple[str, str]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    last_err = None
    for model in DEEPSEEK_MODELS:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 8192,
            "stream": False,
        }
        try:
            resp = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=180
            )
            if resp.status_code != 200:
                last_err = f"{model}: HTTP {resp.status_code} {resp.text[:300]}"
                log(f"  DeepSeek {model} 失败：{last_err}")
                continue
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            log(f"  DeepSeek 调用成功，模型：{model}")
            return content, model
        except Exception as e:  # noqa: BLE001
            last_err = f"{model}: {e}"
            log(f"  DeepSeek {model} 异常：{str(e)[:200]}")

    raise RuntimeError(f"DeepSeek 调用失败：{last_err}")


def build_prompt(top: pd.DataFrame, latest: str) -> str:
    lines = []
    lines.append(
        "你是一位同时精通 ETF 投资与缠论（缠中说禅）技术分析的资深分析师。"
        "下面是基于 Tushare 行情、按月线多头 + 周线回踩条件筛选出的 A 股 ETF 前 "
        f"{len(top)} 名。数据截止交易日：{latest}。\n"
    )
    lines.append("筛选条件：")
    lines.append("1) 月线多头：最新月线收盘价同时高于 5/10/20 月均线（不要求均线多头排列）；")
    lines.append("2) 周线条件：最近 3 个交易日中，至少一天收盘价同时高于 5 周线、10 周线，"
                 "但低于 20 周线。")
    lines.append("说明：月线量能不是筛选条件，下面的量能数据仅作参考。\n")
    lines.append("候选明细（价格均为前复权）：")

    for i, r in top.iterrows():
        if int(r["vol_months"]) > 0:
            vol_line = (
                f"月线量能（仅参考）：月均量对数斜率 {r['vol_slope_pct'] * 100:.1f}%/月，"
                f"R²={r['vol_r2']:.2f}，近1/3期均量比 {r['vol_recent_early']:.2f}，"
                f"最大/中位量比 {r['vol_max_median']:.2f}（统计 {r['vol_months']} 个月）"
            )
        else:
            vol_line = "月线量能（仅参考）：当前月线波段过短，样本不足"
        lines.append(
            f"\n[{i + 1}] {r['name']}（{r['ts_code']}，{r['trade_type']}，"
            f"{r['trade_type_reason']}）\n"
            f"  - 最新价：{r['close']:.3f}，当日涨跌：{r['pct_chg']:.2f}%\n"
            f"  - 月线 MA5/10/20：{r['monthly_ma5']:.3f} / {r['monthly_ma10']:.3f} / "
            f"{r['monthly_ma20']:.3f}；收盘较 MA20 高 {r['close_vs_ma20'] * 100:.1f}%\n"
            f"  - 月线波段：{r['rally_start']} 起涨，至今 {r['rally_months']} 个月，"
            f"区间涨幅 {r['rally_return'] * 100:.1f}%\n"
            f"  - {vol_line}\n"
            f"  - 周线条件：最近 {r['weekly_checked_days']} 个交易日中有 "
            f"{r['weekly_hit_days']} 天命中“收盘>5周线且>10周线且<20周线”；\n"
            f"    命中日 {r['weekly_hit_date']}：收盘 {r['weekly_hit_close']:.3f}，"
            f"周线 MA5/10/20 = {r['weekly_hit_ma5']:.3f} / {r['weekly_hit_ma10']:.3f} / "
            f"{r['weekly_hit_ma20']:.3f}\n"
            f"    最新 {r['weekly_cur_date']}：收盘 {r['weekly_cur_close']:.3f}，"
            f"周线 MA5/10/20 = {r['weekly_cur_ma5']:.3f} / {r['weekly_cur_ma10']:.3f} / "
            f"{r['weekly_cur_ma20']:.3f}\n"
            f"  - 近20日日均成交额：{r['avg_amount_20d_yi']:.2f} 亿元；"
            f"量化综合分：{r['score']:.1f}/100"
        )

    lines.append(
        "\n请用中文输出一份结构化分析报告，要求：\n"
        f"1. 【总体结论】先用 150 字概括这 {len(top)} 只 ETF 的整体结构特征与市场含义；\n"
        "2. 【逐只分析】按下面维度逐只分析（每只 120~180 字）：\n"
        "   (a) 月线级别趋势是否健康，属于“启动初期 / 主升中段 / 高位加速”哪一段；"
        "量能仅作参考，若明显缩量需提示量价背离风险；\n"
        "   (b) 周线条件的技术含义：股价站上 5/10 周线但仍低于 20 周线，"
        "判断这是健康的周线级别回踩修复，还是弱势反弹；结合股价与 20 周线的距离评估突破难度；\n"
        "   (c) 关键观察位：周线 MA5、MA10、MA20 的具体价格，"
        "以及重新站上 20 周线 / 跌破 10 周线分别意味着什么；\n"
        "   (d) T+0 或 T+1 对交易节奏的影响；\n"
        f"3. 【优先级排序】给出 1~{len(top)} 名的排序表，并说明前 3 名入选理由"
        "（可参考量化分，但要结合月线与周线结构修正）；\n"
        "4. 【操作与风控】分别给出“偏进攻 / 偏稳健”两种配置思路，说明关注区间、"
        "失效条件（如跌破周线 MA10 或 20 周线迟迟不能收复）与仓位建议；\n"
        f"5. 【风险提示】指出这 {len(top)} 只中结构最弱、最需要回避的 1~2 只及原因；"
        "并声明本文仅为技术研究，不构成投资建议。\n"
        "注意：不要编造数据，所有价格引用必须来自上面的明细。"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 九、主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="A股 ETF 月线+周线选基")
    parser.add_argument("--outdir", default="/workspace/output")
    parser.add_argument("--lookback-months", type=int, default=36)
    parser.add_argument("--daily-years", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--no-adj", action="store_true")
    parser.add_argument("--skip-ds", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for fn in os.listdir(args.outdir):
        if fn.startswith("etf_top") or fn in (
            "deepseek_analysis.md", "deepseek_prompt.txt"
        ):
            try:
                os.remove(os.path.join(args.outdir, fn))
            except OSError:
                pass

    api = Api(TUSHARE_TOKEN)

    universe = load_etf_universe(api)
    universe_path = os.path.join(args.outdir, "etf_universe_t0_t1.csv")
    universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
    n_t0 = int((universe["trade_type"] == "T+0").sum())
    log(f"ETF 股票池：{len(universe)} 只（T+0 {n_t0} / T+1 {len(universe) - n_t0}）")

    latest = detect_latest_trade_date(api)
    log(f"数据最新交易日：{latest}")

    candidates = prefilter_by_monthly_ma(
        api, universe, latest, lookback_months=args.lookback_months
    )
    candidates_path = os.path.join(args.outdir, "etf_monthly_prefilter.csv")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")

    before = len(candidates)
    candidates = candidates[
        ~candidates["name"].astype(str).str.contains("货币", na=False)
    ].copy()
    log(f"剔除货币ETF {before - len(candidates)} 只，参与选基 {len(candidates)} 只")

    if args.max_candidates > 0:
        candidates = candidates.head(args.max_candidates).copy()

    cal = api.call(
        api.pro.trade_cal,
        exchange="SSE",
        start_date=latest,
        end_date=(datetime.strptime(latest, "%Y%m%d") + timedelta(days=15)).strftime("%Y%m%d"),
        is_open="1",
        fields="cal_date",
    )
    if cal is None or cal.empty:
        future_days = []
    else:
        future_days = sorted(d for d in cal["cal_date"].astype(str) if d > latest)
    month_complete = (not future_days) or future_days[0][:6] != latest[:6]

    start_dt = datetime.strptime(latest, "%Y%m%d") - timedelta(days=int(args.daily_years * 365.25))
    start_date = start_dt.strftime("%Y%m%d")
    log(f"开始精算 {len(candidates)} 只候选（日线起点 {start_date}，复权={not args.no_adj}）...")

    results = []
    t0 = time.time()
    for i, row in enumerate(candidates.itertuples(index=False), 1):
        row = row._asdict()
        try:
            rec = evaluate_etf(
                api, row, start_date, latest,
                use_adj=not args.no_adj, month_complete=month_complete,
            )
            if rec is not None:
                results.append(rec)
        except Exception as e:  # noqa: BLE001
            log(f"  ! {row['ts_code']} 处理失败：{str(e)[:150]}")
        if i % 20 == 0 or i == len(candidates):
            log(f"  进度 {i}/{len(candidates)}，已入选 {len(results)}，"
                f"耗时 {time.time() - t0:.0f}s")

    if not results:
        log(
            f"本次 0 只：月线均线预筛 {len(candidates)} 只，最终没有 ETF 同时满足"
            f"“月线多头 + 最近3个交易日周线条件”。"
        )
        log("这是有效的筛选结果，不调用 DeepSeek；已写入空的 etf_selection_all.csv。")
        pd.DataFrame().to_csv(
            os.path.join(args.outdir, "etf_selection_all.csv"), index=False
        )
        return

    sel = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    sel_path = os.path.join(args.outdir, "etf_selection_all.csv")
    sel.to_csv(sel_path, index=False, encoding="utf-8-sig")
    log(f"入选 {len(sel)} 只，全部结果已保存：{sel_path}")

    top = sel.head(args.top).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top_path = os.path.join(args.outdir, "etf_top10.csv")
    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    log(f"前 {len(top)} 只已保存：{top_path}")

    log(f"入选 {len(sel)} 只，前 {len(top)} 只：")
    for r in top.itertuples():
        log(
            f"  {r.rank:>2}. {r.name}({r.ts_code}) {r.trade_type} "
            f"score={r.score:.1f} 周线命中={r.weekly_hit_days}/{r.weekly_checked_days}天"
        )

    if args.skip_ds:
        log("已跳过 DeepSeek 分析（--skip-ds）。")
        return

    prompt = build_prompt(top, latest)
    prompt_path = os.path.join(args.outdir, "deepseek_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    log(f"调用 DeepSeek 分析前 {len(top)} 只 ETF ...")
    content, model = call_deepseek(prompt)
    md_path = os.path.join(args.outdir, "deepseek_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ETF DeepSeek 分析（模型：{model}）\n\n")
        f.write(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 数据截止：{latest}\n")
        f.write(f"- 选股条件：月线多头 + 最近3日周线回踩条件（量能仅参考）\n\n")
        f.write("---\n\n")
        f.write(content)
    log(f"DeepSeek 分析已保存：{md_path}")


if __name__ == "__main__":
    main()
