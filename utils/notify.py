"""Feishu (Lark) notification client — webhook + app mode.

Setup
-----
Webhook 模式 (简单):
    1. 飞书群 → 设置 → 群机器人 → 添加自定义机器人
    2. 复制 Webhook URL
    3. export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"

App 模式 (功能更强，可发到任意群/人):
    1. 飞书开放平台 → 创建企业自建应用 → 获取 appId / appSecret
    2. 给应用开启"获取群组信息"和"发送消息"权限
    3. export FEISHU_APP_ID="cli_xxx"
       export FEISHU_APP_SECRET="xxx"
       export FEISHU_CHAT_ID="oc_xxx"  (群聊ID，从飞书开发者后台看)

Usage
-----
    from utils.notify import Notifier

    nf = Notifier()                              # 自动读取环境变量
    nf.text("AAPL trend_follower 买入信号 $195")  # 简单文本
    nf.signal_card(signals)                      # 结构化信号卡片
    nf.trade_card(order)                         # 成交通知
    nf.error("数据源连接失败")                     # 错误告警
"""

import atexit
import json
import logging as _logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feishu card color palette
# ---------------------------------------------------------------------------

COLOR = {
    "buy": "green",
    "sell": "red",
    "info": "blue",
    "warn": "yellow",
    "error": "red",
    "title": "turquoise",
}


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier:
    """Feishu notification client — auto-detects webhook vs app mode.

    Priority: webhook URL > app credentials > dry-run

    Webhook mode: POST directly to webhook URL (no auth needed)
    App mode:     POST to message API with tenant_access_token

    Parameters
    ----------
    url : str | None
        Webhook URL.  If None, reads FEISHU_WEBHOOK from env.
    app_id : str | None
        Feishu app ID.  If None, reads FEISHU_APP_ID from env.
    app_secret : str | None
        Feishu app secret.  If None, reads FEISHU_APP_SECRET from env.
    chat_id : str | None
        Target chat ID for app mode.  Reads FEISHU_CHAT_ID from env.
    dry_run : bool
        If True, print messages instead of sending.

    Environment
    -----------
    FEISHU_WEBHOOK    : webhook URL (takes priority if set)
    FEISHU_APP_ID     : app ID for app mode
    FEISHU_APP_SECRET : app secret for app mode
    FEISHU_CHAT_ID    : target chat ID for app mode
    """

    _token: Optional[str] = None
    _token_expires: float = 0

    def __init__(
        self,
        url: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        chat_id: Optional[str] = None,
        dry_run: bool = False,
        async_mode: bool = True,
    ):
        self.url = url or os.getenv("FEISHU_WEBHOOK", "")
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "")
        self.chat_id = chat_id or os.getenv("FEISHU_CHAT_ID", "")
        self.dry_run = dry_run
        self._async = async_mode
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._worker: Optional[threading.Thread] = None

        if self.url:
            self._mode = "webhook"
        elif self.app_id and self.app_secret:
            self._mode = "app"
        else:
            self._mode = "none"
            if not dry_run:
                logger.warning("飞书通知未配置 — 设置 FEISHU_WEBHOOK 或 FEISHU_APP_ID+FEISHU_APP_SECRET")

        if self._async and self.available:
            self._start_worker()

    @property
    def available(self) -> bool:
        return self._mode != "none" or self.dry_run

    # ------------------------------------------------------------------
    # Async worker
    # ------------------------------------------------------------------

    def _start_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._consume, name="notify-worker", daemon=True,
        )
        self._worker.start()
        atexit.register(self.stop)

    def _consume(self):
        while True:
            try:
                payload = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.enqueue(payload)
            except Exception:
                logger.exception("通知消费异常")

    def stop(self):
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)

    def enqueue(self, payload: dict) -> bool:
        """Enqueue a payload for async delivery. Falls back to sync if async_mode=False."""
        if not self._async:
            return self._send(payload)
        if not self.available:
            return False
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning("通知队列已满(%d)，丢弃消息", self._queue.maxsize)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def text(self, content: str) -> bool:
        """Send a plain-text message.  Max 20KB."""
        payload = {
            "msg_type": "text",
            "content": {"text": content},
        }
        return self.enqueue(payload)

    _strat_cn = {
        "enhanced_macd": "增强MACD",
        "trend_follower": "趋势跟踪",
        "weekly_macd": "周线MACD",
        "weekly_macd_kdj": "周线KDJ+MACD",
    }

    def _strat_label(self, key: str) -> str:
        return self._strat_cn.get(key, key)

    def signal_card(self, signals: List[dict], scan_date: str = "") -> bool:
        """Send a rich card summarising today's signals.

        Parameters
        ----------
        signals : list[dict]
            Each dict has keys: symbol, strategy, signal (1/-1), price
        scan_date : str
            YYYY-MM-DD
        """
        buys = [s for s in signals if s["signal"] == 1]
        sells = [s for s in signals if s["signal"] == -1]

        if not buys and not sells:
            return self.text(f"[{scan_date}] 今日无买入/卖出信号")

        elements = []
        if buys:
            elements.append(self._mk_field(
                f"买入信号 ({len(buys)})",
                "\n".join(f"{s['symbol']}  {self._strat_label(s['strategy'])}  ${s['price']:.2f}"
                          for s in buys),
            ))
        if sells:
            elements.append(self._mk_field(
                f"卖出信号 ({len(sells)})",
                "\n".join(f"{s['symbol']}  {self._strat_label(s['strategy'])}  ${s['price']:.2f}"
                          for s in sells),
            ))

        card = self._mk_card(
            title=f"每日回溯 — {scan_date}",
            color=COLOR["buy"] if buys else COLOR["sell"],
            elements=elements,
            footer=f"共 {len(signals)} 个策略-标的组合  |  {datetime.now().strftime('%H:%M')}",
        )
        return self._send({"msg_type": "interactive", "card": card})

    def trade_card(self, order) -> bool:
        """Send a trade fill notification."""
        side = "买入" if order.side.value == "BUY" else "卖出"
        color = COLOR["buy"] if order.side.value == "BUY" else COLOR["sell"]
        status = "已成交" if order.status.value == "FILLED" else order.status.value

        elements = [
            self._mk_field("标的", order.symbol),
            self._mk_field("方向", f"{side} {order.filled_qty}股"),
            self._mk_field("价格", f"${order.avg_fill_price:.2f}"),
            self._mk_field("状态", status),
            self._mk_field("订单ID", order.order_id),
        ]
        card = self._mk_card(
            title=f"交易通知 — {side} {order.symbol}",
            color=color,
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    def error(self, message: str, context: str = "") -> bool:
        """Send an error / alert notification."""
        elements = [self._mk_field("错误", message)]
        if context:
            elements.append(self._mk_field("上下文", context))
        card = self._mk_card(
            title="系统告警",
            color=COLOR["error"],
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    def daily_summary(self, buy_count: int, sell_count: int, total: int,
                      account_equity: float = 0, positions: int = 0) -> bool:
        """End-of-day summary card."""
        elements = [
            self._mk_field("买入信号", str(buy_count)),
            self._mk_field("卖出信号", str(sell_count)),
            self._mk_field("扫描组合数", str(total)),
        ]
        if account_equity > 0:
            elements.append(self._mk_field("账户权益", f"${account_equity:,.0f}"))
            elements.append(self._mk_field("持仓数", str(positions)))

        card = self._mk_card(
            title=f"每日回溯完成 — {datetime.now().strftime('%Y-%m-%d')}",
            color=COLOR["info"],
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    # ------------------------------------------------------------------
    # Internal — send
    # ------------------------------------------------------------------

    def _send(self, payload: dict) -> bool:
        if self.dry_run:
            logger.info("[DRY-RUN] 飞书通知: %s", json.dumps(payload, ensure_ascii=False)[:200])
            return True

        if self._mode == "webhook":
            return self._send_webhook(payload)
        elif self._mode == "app":
            return self._send_app(payload)
        return False

    def _send_webhook(self, payload: dict) -> bool:
        try:
            r = requests.post(self.url, json=payload, timeout=10)
            if r.status_code == 200:
                resp = r.json()
                if resp.get("code") == 0:
                    return True
                logger.warning("飞书 webhook 返回错误: %s", resp.get("msg", "unknown"))
            else:
                logger.warning("飞书 webhook HTTP %d: %s", r.status_code, r.text[:100])
        except Exception as e:
            logger.error("飞书 webhook 发送失败: %s", e)
        return False

    def _send_app(self, payload: dict) -> bool:
        """Send via Feishu message API using app credentials."""
        token = self._get_app_token()
        if not token:
            return False
        if not self.chat_id:
            logger.warning("FEISHU_CHAT_ID 未设置，无法发送 app 模式消息")
            return False

        # Determine message type
        msg_type = payload.get("msg_type", "text")
        content = payload.get("content", payload.get("card"))

        body = {
            "receive_id": self.chat_id,
            "msg_type": msg_type,
            "content": json.dumps(content) if isinstance(content, dict) else content,
        }

        try:
            r = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params={"receive_id_type": "chat_id"},
                json=body,
                timeout=10,
            )
            if r.status_code == 200:
                resp = r.json()
                if resp.get("code") == 0:
                    return True
                logger.warning("飞书 app 返回错误: %s", resp.get("msg", "unknown"))
            else:
                logger.warning("飞书 app HTTP %d: %s", r.status_code, r.text[:100])
        except Exception as e:
            logger.error("飞书 app 发送失败: %s", e)
        return False

    def _get_app_token(self) -> Optional[str]:
        """Get or refresh tenant_access_token."""
        now = time.time()
        if self._token and now < self._token_expires - 60:
            return self._token

        try:
            r = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == 0:
                    self._token = data["tenant_access_token"]
                    self._token_expires = now + data.get("expire", 7200)
                    return self._token
                logger.warning("获取飞书 token 失败: %s", data.get("msg"))
            else:
                logger.warning("飞书 token HTTP %d", r.status_code)
        except Exception as e:
            logger.error("飞书 token 请求异常: %s", e)
        return None

    @staticmethod
    def _mk_card(title: str, color: str, elements: List[dict],
                 footer: str = "") -> dict:
        card = {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,
            },
            "elements": elements,
        }
        if footer:
            card["elements"].append({
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": footer}],
            })
        return card

    @staticmethod
    def _mk_field(label: str, value: str) -> dict:
        return {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**{label}**\n{value}"},
        }


# ---------------------------------------------------------------------------
# Logging → Notification bridge
# ---------------------------------------------------------------------------


class NotifyLogHandler(_logging.Handler):
    """Bridge Python logging ERROR+ records to Notifier via async queue."""

    def __init__(self, notifier: Optional[Notifier] = None, level: int = _logging.ERROR):
        super().__init__(level=level)
        self._notifier = notifier or Notifier()
        self.setFormatter(_logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: _logging.LogRecord):
        try:
            msg = self.format(record)
            self._notifier.error(msg, context=record.name)
        except Exception:
            self.handleError(record)


def install_notify_log_handler(
    notifier: Optional[Notifier] = None,
    level: int = _logging.ERROR,
    fmt: str = "[%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Install NotifyLogHandler on the root logger.  Idempotent."""
    notifier = notifier or Notifier()
    root = _logging.getLogger()
    for h in root.handlers:
        if isinstance(h, NotifyLogHandler):
            return
    handler = NotifyLogHandler(notifier, level=level)
    handler.setFormatter(_logging.Formatter(fmt))
    root.addHandler(handler)
