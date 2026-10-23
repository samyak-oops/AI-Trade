"""
tools/broker.py – Angel One SmartAPI Integration & Paper Trading Engine.
Supports live order placement, portfolio fetching, and paper trading mode.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pyotp
from SmartApi import SmartConnect

from core.config import get_settings

logger = logging.getLogger(__name__)


class AngelBroker:
    """Angel One SmartAPI Client with automated TOTP session management."""

    def __init__(self) -> None:
        self.cfg = get_settings()
        self._smart_api: Optional[SmartConnect] = None
        self._session_data: Dict[str, Any] = {}

    def connect(self) -> bool:
        """Authenticate with Angel One using credentials and TOTP."""
        if not (self.cfg.angel_api_key and self.cfg.angel_client_id and self.cfg.angel_password and self.cfg.angel_totp_secret):
            logger.warning("Angel One credentials incomplete in .env.")
            return False

        try:
            totp = pyotp.TOTP(self.cfg.angel_totp_secret.strip()).now()
            self._smart_api = SmartConnect(api_key=self.cfg.angel_api_key.strip())
            session = self._smart_api.generateSession(
                self.cfg.angel_client_id.strip(),
                self.cfg.angel_password.strip(),
                totp,
            )
            if session and session.get("status"):
                self._session_data = session.get("data", {})
                logger.info("Angel One SmartAPI session established successfully.")
                return True
            else:
                logger.error("Angel One authentication failed: %s", session.get("message"))
                return False
        except Exception as e:
            logger.error("Exception during Angel One login: %s", e)
            return False

    @property
    def is_connected(self) -> bool:
        return self._smart_api is not None and bool(self._session_data)

    def get_profile(self) -> Dict[str, Any]:
        """Fetch client profile from Angel One."""
        if not self.is_connected and not self.connect():
            return {"error": "Not connected to Angel One"}
        try:
            return self._smart_api.getProfile(self._session_data.get("refreshToken"))
        except Exception as e:
            return {"error": str(e)}

    def get_rms_limits(self) -> Dict[str, Any]:
        """Fetch available margin / funds."""
        if not self.is_connected and not self.connect():
            return {"net": self.cfg.default_capital, "paper_mode": True}
        try:
            res = self._smart_api.rmsLimit()
            return res.get("data", {}) if res and res.get("status") else {}
        except Exception as e:
            logger.error("Failed to fetch RMS limits: %s", e)
            return {"net": self.cfg.default_capital, "error": str(e)}

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Fetch user's long-term holdings."""
        if not self.is_connected and not self.connect():
            return []
        try:
            res = self._smart_api.holding()
            return res.get("data", []) if res and res.get("status") else []
        except Exception as e:
            logger.error("Failed to fetch holdings: %s", e)
            return []

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch user's open day positions."""
        if not self.is_connected and not self.connect():
            return []
        try:
            res = self._smart_api.position()
            return res.get("data", []) if res and res.get("status") else []
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

    def place_order(
        self,
        symbol: str,
        token: str,
        transaction_type: str,  # BUY or SELL
        quantity: int,
        order_type: str = "MARKET",  # MARKET or LIMIT
        price: float = 0.0,
        exchange: str = "NSE",
        product_type: str = "DELIVERY",  # DELIVERY or INTRADAY
    ) -> Dict[str, Any]:
        """Place an order (simulated in PAPER_TRADING mode, real otherwise)."""
        if self.cfg.paper_trading:
            logger.info("[PAPER TRADING] Order executed: %s %d %s @ %s", transaction_type, quantity, symbol, price)
            return {
                "status": True,
                "paper_trading": True,
                "message": f"Simulated {transaction_type} of {quantity} {symbol} successfully recorded.",
                "data": {
                    "orderid": f"PAPER-{symbol}-{quantity}",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "type": transaction_type,
                },
            }

        if not self.is_connected and not self.connect():
            return {"status": False, "message": "Failed to connect to Angel One broker."}

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": str(token),
            "transactiontype": transaction_type.upper(),
            "exchange": exchange.upper(),
            "ordertype": order_type.upper(),
            "producttype": product_type.upper(),
            "duration": "DAY",
            "price": str(price) if order_type.upper() == "LIMIT" else "0",
            "quantity": str(quantity),
        }

        try:
            res = self._smart_api.placeOrder(order_params)
            return res
        except Exception as e:
            logger.error("Order placement exception: %s", e)
            return {"status": False, "message": str(e)}


# Singleton broker instance
_broker_instance: Optional[AngelBroker] = None


def get_broker() -> AngelBroker:
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = AngelBroker()
    return _broker_instance
