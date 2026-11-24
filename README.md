# stock-etf

A 股 ETF / 股票月线 + 周线技术面选股脚本。

## 文件

- `etf_chan_selector.py`：ETF 选基，含 T+0 / T+1 分类
- `stock_chan_selector.py`：A 股股票选股

## 筛选逻辑（全部为月线级别）

1. 最新一根月线 MA5 > MA10；
2. 前 5 根月线中，至少有一根 MA5 < MA10（最近 5 个月内出现金叉）；
3. 最近 3 根月线，MA5 的斜率大于 MA10 的斜率；
4. 量能仅作参考，不参与筛选；
5. 按“金叉月龄 + MA5/MA10 开口 + 斜率加速 + 距 MA20 + 距 12 个月低点 + 流动性”
   综合打分，取前 10 只调用 DeepSeek 做结构化分析。

## 环境变量

不要把 token 写进代码。运行前设置：

```bash
export TUSHARE_TOKEN="你的 tushare token"
export DEEPSEEK_API_KEY="你的 deepseek token"
```

可选：

```bash
export DEEPSEEK_MODEL="deepseek-flash"   # 默认 deepseek-flash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

## 运行

```bash
python3 etf_chan_selector.py
python3 stock_chan_selector.py
```

快捷参数：

```bash
python3 etf_chan_selector.py --skip-ds
python3 stock_chan_selector.py --skip-ds
python3 stock_chan_selector.py --outdir /workspace/output_stock
```

依赖：`pandas`、`numpy`、`requests`、`tushare`。
