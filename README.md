# stock-etf

A 股 ETF / 股票月线 + 周线技术面选股脚本。

## 文件

- `etf_chan_selector.py`：ETF 选基，含 T+0 / T+1 分类
- `stock_chan_selector.py`：A 股股票选股

## 筛选逻辑

1. 月线多头：最新月线收盘价同时大于 5/10/20 月均线；
2. 周线条件：最近 3 个交易日中，至少一天收盘价同时大于 5 周线、10 周线，
   但小于 20 周线；
3. 量能仅作参考，不参与筛选；
4. 综合打分后取前 10 只，调用 DeepSeek 做结构化分析。

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
