#!/bin/bash
# 启动实盘守护进程 — FutuBroker 模拟盘 + 飞书通知
cd "$(dirname "$0")"
pipenv run python live_trader.py \
  --broker futu \
  --futu-host 127.0.0.1 \
  --futu-port 11111 \
  --daemon \
  --interval 5 \
  --notify \
  --all-day
