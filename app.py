
import os
from flask import Flask, render_template, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")

def get_client():
    return Client(API_KEY, SECRET_KEY)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/futures")
def futures_data():
    try:
        client = get_client()

        # 선물 계좌 잔고 정보
        account = client.futures_account()

        total_balance = float(account["totalWalletBalance"])
        unrealized_pnl = float(account["totalUnrealizedProfit"])
        available_balance = float(account["availableBalance"])
        margin_balance = float(account["totalMarginBalance"])
        margin_ratio = float(account.get("totalMaintMargin", 0))

        # 현재 포지션 (진입 수량 0 이상인 것만)
        positions = []
        for pos in account["positions"]:
            amt = float(pos["positionAmt"])
            if amt != 0:
                entry_price = float(pos["entryPrice"])
                mark_price = float(pos.get("markPrice", 0))
                pnl = float(pos["unrealizedProfit"])
                leverage = int(pos.get("leverage", 1))
                side = "LONG" if amt > 0 else "SHORT"

                # 수익률 계산
                if entry_price > 0:
                    if side == "LONG":
                        pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * leverage
                    else:
                        pnl_pct = ((entry_price - mark_price) / entry_price) * 100 * leverage
                else:
                    pnl_pct = 0

                positions.append({
                    "symbol": pos["symbol"],
                    "side": side,
                    "amount": abs(amt),
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "unrealized_pnl": pnl,
                    "pnl_pct": round(pnl_pct, 2),
                    "leverage": leverage,
                    "notional": abs(float(pos.get("notional", amt * mark_price))),
                })

        # 자금 여유도 판단
        if available_balance >= margin_balance * 0.5:
            fund_status = "여유"
            fund_color = "green"
        elif available_balance >= margin_balance * 0.2:
            fund_status = "보통"
            fund_color = "amber"
        else:
            fund_status = "위험"
            fund_color = "red"

        return jsonify({
            "success": True,
            "account": {
                "total_balance": round(total_balance, 2),
                "margin_balance": round(margin_balance, 2),
                "available_balance": round(available_balance, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "fund_status": fund_status,
                "fund_color": fund_color,
            },
            "positions": positions,
        })

    except BinanceAPIException as e:
        return jsonify({"success": False, "error": f"Binance API 오류: {e.message}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
