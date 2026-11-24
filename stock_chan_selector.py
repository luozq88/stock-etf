#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股股票：月线 MA5/MA10 金叉 + 斜率加速 + DeepSeek 分析

选股条件（全部为月线级别）：
1) 最新一根月线 MA5 > MA10；
2) 前 5 根月线中，至少有一根 MA5 < MA10（最近 5 个月内出现金叉）；
3) 最近 3 根月线，MA5 的斜率大于 MA10 的斜率；
4) 综合打分后取前 10 只，调用 DeepSeek（flash 版）做结构化分析。

说明：该结构用于捕捉“月线刚从底部启动”；量能不再作为硬性条件，仅作参考。
数据源：Tushare Pro（stock_basic / daily / adj_factor / trade_cal）
复用 etf_chan_selector.py 中的月线、量能计算函数。
用法：
    python3 stock_chan_selector.py
    python3 stock_chan_selector.py --skip-ds
    python3 stock_chan_selector.py --outdir /workspace/output_stock
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import etf_chan_selector as base


def log(msg: str) -> None:
    base.log(msg)


# ----------------------------------------------------------------------------
# 一、A 股股票池
# ----------------------------------------------------------------------------
def load_stock_universe(api: base.Api) -> pd.DataFrame:
    df = api.call(
        api.pro.stock_basic,
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if df is None or df.empty:
        raise RuntimeError("stock_basic 返回为空")

    df = df.copy()
    df["name"] = df["name"].astype(str)
    df["list_date"] = df["list_date"].astype(str)
    # 剔除 ST / *ST / 退市整理
    df = df[~df["name"].str.contains("ST|退", regex=True, na=False)].copy()
    # 剔除上市不足 24 个月的股票（月线 MA20 至少需要 20 个月数据）
    cutoff = (datetime.now() - timedelta(days=24 * 31)).strftime("%Y%m%d")
    df = df[df["list_date"] <= cutoff].copy()
    df["trade_type"] = "T+1"
    df["trade_type_reason"] = "A股股票（T+1）"
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 二、最新交易日与月末快照
# ----------------------------------------------------------------------------
def detect_latest_trade_date(api: base.Api, lookback_days: int = 400) -> str:
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
        df = api.call(api.pro.daily, trade_date=d, fields="ts_code,close")
        if df is not None and len(df) > 1000:
            return d
    raise RuntimeError("未找到有行情的最近交易日")


def prefilter_by_monthly_ma(api: base.Api, universe: pd.DataFrame, latest: str,
                            lookback_months: int = 36) -> pd.DataFrame:
    """用每月最后一个交易日的收盘价快速预筛：收盘 > MA5/10/20。"""
    start_dt = datetime.strptime(latest, "%Y%m%d") - timedelta(days=lookback_months * 31 + 40)
    month_ends = base.month_end_trade_dates(
        api, start_dt.strftime("%Y%m%d"), latest
    )[-lookback_months:]
    log(f"预筛：拉取 {len(month_ends)} 个月末交易日收盘价 ...")

    frames = []
    last_snapshot = None
    for i, d in enumerate(month_ends, 1):
        df = api.call(
            api.pro.daily,
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
        raise RuntimeError("月末行情为空")

    px = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["ts_code", "month"], keep="last")
        .pivot_table(index="ts_code", columns="month", values="close", aggfunc="last")
        .sort_index(axis=1)
    )
    log(f"  收盘价矩阵：{px.shape[0]} 只股票 × {px.shape[1]} 个月")

    keep = []
    for code, row in px.iterrows():
        if base.monthly_cross_signal(row.values) is not None:
            keep.append(code)

    log(f"  月线 MA5/MA10 金叉预筛通过：{len(keep)} 只")
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
# 三、单只股票日线（前复权）
# ----------------------------------------------------------------------------
def fetch_daily(api: base.Api, ts_code: str, start_date: str, end_date: str,
                use_adj: bool = True) -> pd.DataFrame | None:
    df = api.call(
        api.pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date
    )
    if df is None or len(df) < 80:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str)

    if use_adj:
        try:
            adj = api.call(
                api.pro.adj_factor,
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
    return df if len(df) >= 80 else None


# ----------------------------------------------------------------------------
# 四、单只股票评估
# ----------------------------------------------------------------------------
def evaluate_stock(api: base.Api, row: dict, start_date: str, latest: str,
                   use_adj: bool, month_complete: bool):
    code = row["ts_code"]
    daily = fetch_daily(api, code, start_date, latest, use_adj=use_adj)
    if daily is None:
        return None

    monthly = base.to_monthly(daily)
    if len(monthly) < base.MIN_MONTHS:
        return None

    close = float(monthly["close"].iloc[-1])
    sig = base.monthly_cross_signal(monthly["close"].values)
    if sig is None:
        return None

    ma5 = sig["ma5"]
    ma10 = sig["ma10"]
    ma20 = float(monthly["close"].iloc[-20:].mean())
    gap = sig["gap"]
    cross_age = sig["cross_age"]
    slope5 = sig["slope5"]
    slope10 = sig["slope10"]
    accel = sig["accel"]

    rally_i = base.find_rally_start(monthly)
    if rally_i is None:
        return None

    # 月线量能不再作为筛选条件，仅作为展示/风险参考
    vt = base.volume_trend(monthly, rally_i, drop_last=not month_complete)
    if vt is None:
        vt = {
            "slope_pct": np.nan,
            "r2": np.nan,
            "recent_early_ratio": np.nan,
            "max_median_ratio": np.nan,
            "months": 0,
            "score": 0.0,
        }

    rally_low = float(monthly["low"].iloc[rally_i])
    rally_months = len(monthly) - rally_i
    rally_ret = close / rally_low - 1.0
    low12 = float(monthly["low"].tail(12).min())
    rally_ret12 = close / low12 - 1.0

    dev20 = close / ma20 - 1.0

    def clip01(x):
        return float(min(1.0, max(0.0, x)))

    # 越接近启动越好：金叉越新、开口适中、加速度越大、离 MA20 和主底部越近
    cross_score = clip01((6.0 - cross_age) / 5.0)
    gap_score = float(np.tanh(gap / 0.03))
    accel_score = float(np.tanh(accel / (abs(slope10) + 0.02)))
    position_score = clip01(1.0 - dev20 / 0.25)
    bottom_score = clip01(1.0 - rally_ret / 1.0)

    amt20 = float(daily["amount"].tail(20).mean())
    amt_yi = amt20 / 1e5
    liq_score = clip01(amt_yi / 2.0)

    total = 100.0 * (
        0.25 * cross_score
        + 0.20 * gap_score
        + 0.20 * accel_score
        + 0.20 * position_score
        + 0.10 * bottom_score
        + 0.05 * liq_score
    )
    pct = float(daily["pct_chg"].iloc[-1]) if "pct_chg" in daily.columns else np.nan

    return {
        "ts_code": code,
        "name": row["name"],
        "industry": row.get("industry", ""),
        "market": row.get("market", ""),
        "area": row.get("area", ""),
        "list_date": row.get("list_date", ""),
        "trade_type": "T+1",
        "close": close,
        "pct_chg": pct,
        "monthly_ma5": ma5,
        "monthly_ma10": ma10,
        "monthly_ma20": ma20,
        "ma5_ma10_gap": gap,
        "cross_age": cross_age,
        "ma5_slope": slope5,
        "ma10_slope": slope10,
        "slope_accel": accel,
        "close_vs_ma20": dev20,
        "rally_start": monthly.index[rally_i].strftime("%Y-%m-%d"),
        "rally_months": rally_months,
        "rally_low": rally_low,
        "rally_return": rally_ret,
        "low_12m": low12,
        "rally_return_12m": rally_ret12,
        "vol_slope_pct": vt["slope_pct"],
        "vol_r2": vt["r2"],
        "vol_recent_early": vt["recent_early_ratio"],
        "vol_max_median": vt["max_median_ratio"],
        "vol_months": vt["months"],
        "avg_amount_20d_yi": amt_yi,
        "gap_score": gap_score,
        "cross_score": cross_score,
        "accel_score": accel_score,
        "position_score": position_score,
        "bottom_score": bottom_score,
        "liquidity_score": liq_score,
        "score": total,
    }


# ----------------------------------------------------------------------------
# 五、DeepSeek 提示词
# ----------------------------------------------------------------------------
def build_prompt(top: pd.DataFrame, latest: str) -> str:
    lines = []
    lines.append(
        "你是一位同时精通 A 股股票投资与技术分析的资深分析师。"
        "下面是基于 Tushare 行情、按“月线 MA5/MA10 金叉 + 斜率加速”"
        f"筛选出的 A 股股票前 {len(top)} 名。数据截止交易日：{latest}。\n"
    )
    lines.append("选股条件（全部为月线级别）：")
    lines.append("1) 最新一根月线 MA5 > MA10；")
    lines.append("2) 前 5 根月线中，至少有一根 MA5 < MA10（即最近 5 个月内出现金叉）；")
    lines.append("3) 最近 3 根月线，MA5 的斜率大于 MA10 的斜率；")
    lines.append("4) 全部为 A 股股票，T+1 交易；涨跌幅限制：主板 10%，"
                 "创业板/科创板 20%，北交所 30%。")
    lines.append("说明：这三条用于捕捉“月线刚从底部启动”的结构；量能仅作参考。\n")
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
            f"\n[{i + 1}] {r['name']}（{r['ts_code']}，{r['industry']}，{r['market']}）\n"
            f"  - 最新价：{r['close']:.3f}，当日涨跌：{r['pct_chg']:.2f}%\n"
            f"  - 月线 MA5/10/20：{r['monthly_ma5']:.3f} / {r['monthly_ma10']:.3f} / "
            f"{r['monthly_ma20']:.3f}；MA5 较 MA10 高 {r['ma5_ma10_gap'] * 100:.2f}%，"
            f"收盘较 MA20 高 {r['close_vs_ma20'] * 100:.1f}%\n"
            f"  - 金叉月龄：约 {r['cross_age']} 个月前 MA5 上穿 MA10；"
            f"最近3月 MA5 斜率 {r['ma5_slope']:.3f}/月，MA10 斜率 {r['ma10_slope']:.3f}/月，"
            f"斜率差 {r['slope_accel']:.3f}\n"
            f"  - 月线波段：{r['rally_start']} 起涨，低点 {r['rally_low']:.3f}，"
            f"至今 {r['rally_months']} 个月，区间涨幅 {r['rally_return'] * 100:.1f}%\n"
            f"  - 12个月低点：{r['low_12m']:.3f}，当前较其高 "
            f"{r['rally_return_12m'] * 100:.1f}%\n"
            f"  - {vol_line}\n"
            f"  - 近20日日均成交额：{r['avg_amount_20d_yi']:.2f} 亿元；"
            f"量化综合分：{r['score']:.1f}/100"
        )

    lines.append(
        "\n请用中文输出一份结构化分析报告，要求：\n"
        "1. 【总体结论】先用 150 字概括这批股票的月线金叉与斜率加速特征；\n"
        "2. 【逐只分析】按下面维度逐只分析（每只 120~180 字）：\n"
        "   (a) 月线金叉的有效性：MA5/MA10 交叉时间、当前开口大小，"
        "结合距 12 个月低点的涨幅，判断是“刚从底部启动”还是已经走了一段；\n"
        "   (b) 斜率加速：最近 3 个月 MA5 与 MA10 的斜率差，"
        "判断处于加速初期、加速中段还是过热阶段；\n"
        "   (c) 关键观察位：月线 MA5、MA10、MA20 的具体价格，"
        "以及最近月线波段低点；跌破 MA10 / MA20 分别意味着什么；\n"
        "   (d) 所属行业与板块属性，以及 10%/20%/30% 涨跌幅限制对交易节奏的影响；\n"
        "   (e) 风险：金叉失败、斜率走平、乖离过大；量能仅作参考，"
        "若明显缩量需提示量价背离；\n"
        f"3. 【优先级排序】给出 1~{len(top)} 名的排序表，并说明前 3 名入选理由"
        "（可参考量化分，但要结合月线结构修正）；\n"
        "4. 【操作与风控】分别给出“偏进攻 / 偏稳健”两种配置思路，说明关注区间、"
        "失效条件（如跌破月线 MA10 或 MA5 重新下穿 MA10）与仓位建议；\n"
        f"5. 【风险提示】指出这 {len(top)} 只中结构最弱、最需要回避的 1~2 只及原因；"
        "并声明本文仅为技术研究，不构成投资建议。\n"
        "注意：不要编造数据，所有价格引用必须来自上面的明细。"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 六、主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="A股股票月线 MA5/MA10 金叉选股")
    parser.add_argument("--outdir", default="/workspace/output_stock")
    parser.add_argument("--lookback-months", type=int, default=36)
    parser.add_argument("--daily-years", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--no-adj", action="store_true")
    parser.add_argument("--skip-ds", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for fn in os.listdir(args.outdir):
        if fn.startswith("stock_top") or fn in (
            "deepseek_stock_analysis.md", "deepseek_stock_prompt.txt"
        ):
            try:
                os.remove(os.path.join(args.outdir, fn))
            except OSError:
                pass

    api = base.Api(base.TUSHARE_TOKEN)

    # 1) 股票池
    universe = load_stock_universe(api)
    universe_path = os.path.join(args.outdir, "stock_universe.csv")
    universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
    log(f"A股股票池：{len(universe)} 只（已剔除 ST/退市/上市不足24个月）")

    # 2) 最新交易日
    latest = detect_latest_trade_date(api)
    log(f"数据最新交易日：{latest}")

    # 3) 月线 MA5/MA10 金叉预筛
    candidates = prefilter_by_monthly_ma(
        api, universe, latest, lookback_months=args.lookback_months
    )
    candidates_path = os.path.join(args.outdir, "stock_monthly_prefilter.csv")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")

    if args.max_candidates > 0:
        candidates = candidates.head(args.max_candidates).copy()

    # 当前月是否已走完
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

    # 4) 逐只精算
    start_dt = datetime.strptime(latest, "%Y%m%d") - timedelta(days=int(args.daily_years * 365.25))
    start_date = start_dt.strftime("%Y%m%d")
    log(f"开始精算 {len(candidates)} 只候选（日线起点 {start_date}，复权={not args.no_adj}）...")

    results = []
    t0 = time.time()
    for i, row in enumerate(candidates.itertuples(index=False), 1):
        row = row._asdict()
        try:
            rec = evaluate_stock(
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
            f"本次 0 只：月线金叉预筛 {len(candidates)} 只，最终没有股票同时满足"
            f"“MA5>MA10 + 近5月内出现过MA5<MA10 + 近3月MA5斜率>MA10斜率”。"
        )
        log("这是有效的筛选结果，不调用 DeepSeek；已写入空的 stock_selection_all.csv。")
        pd.DataFrame().to_csv(
            os.path.join(args.outdir, "stock_selection_all.csv"), index=False
        )
        return

    sel = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    sel_path = os.path.join(args.outdir, "stock_selection_all.csv")
    sel.to_csv(sel_path, index=False, encoding="utf-8-sig")
    log(f"入选 {len(sel)} 只，全部结果已保存：{sel_path}")

    top = sel.head(args.top).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top_path = os.path.join(args.outdir, "stock_top10.csv")
    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    log(f"前 {len(top)} 只已保存：{top_path}")

    log(f"入选 {len(sel)} 只，前 {len(top)} 只：")
    for r in top.itertuples():
        log(
            f"  {r.rank:>2}. {r.name}({r.ts_code}) {r.industry} "
            f"score={r.score:.1f} 金叉月龄={r.cross_age} "
            f"MA5/MA10={r.monthly_ma5:.3f}/{r.monthly_ma10:.3f}"
        )

    # 5) DeepSeek
    if args.skip_ds:
        log("已跳过 DeepSeek 分析（--skip-ds）。")
        return

    prompt = build_prompt(top, latest)
    prompt_path = os.path.join(args.outdir, "deepseek_stock_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    log(f"调用 DeepSeek 分析前 {len(top)} 只股票 ...")
    content, model = base.call_deepseek(
        prompt,
        system_prompt="你是严谨的 A 股股票技术分析师，只基于给定数据推理，不编造数据。",
    )
    md_path = os.path.join(args.outdir, "deepseek_stock_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# A股股票 DeepSeek 分析（模型：{model}）\n\n")
        f.write(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 数据截止：{latest}\n")
        f.write(f"- 选股条件：月线 MA5/MA10 金叉 + 近3月斜率加速（量能仅参考）\n\n")
        f.write("---\n\n")
        f.write(content)
    log(f"DeepSeek 分析已保存：{md_path}")


if __name__ == "__main__":
    main()
