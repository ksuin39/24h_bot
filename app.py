import os
import hmac
import hashlib
import time
import requests
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

BASE_URL = "https://fapi.binance.com"

def sign(params, secret):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + f"&signature={sig}"

def api_get(path, api_key, secret_key, params=None):
    if params is None:
        params = {}
    params["timestamp"] = int(time.time() * 1000)
    query = sign(params, secret_key)
    url = f"{BASE_URL}{path}?{query}"
    headers = {"X-MBX-APIKEY": api_key}
    res = requests.get(url, headers=headers, timeout=10)
    return res.json()

def is_logged_in():
    return "api_key" in session and "secret_key" in session

@app.route("/")
def index():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        api_key = data.get("api_key", "").strip()
        secret_key = data.get("secret_key", "").strip()

        if not api_key or not secret_key:
            return jsonify({"success": False, "error": "API 키와 시크릿 키를 모두 입력해주세요."})

        try:
            result = api_get("/fapi/v2/balance", api_key, secret_key)
            if isinstance(result, dict) and result.get("code"):
                return jsonify({"success": False, "error": f"API 오류 {result['code']}: {result.get('msg', '')}"})
            session["api_key"] = api_key
            session["secret_key"] = secret_key
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": f"연결 실패: {str(e)}"})

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/api/futures")
def futures_data():
    if not is_logged_in():
        return jsonify({"success": False, "error": "로그인이 필요합니다."}), 401

    try:
        api_key = session["api_key"]
        secret_key = session["secret_key"]

        account = api_get("/fapi/v2/account", api_key, secret_key)

        if isinstance(account, dict) and account.get("code"):
            return jsonify({"success": False, "error": account.get("msg", "API 오류")})

        total_balance = float(account["totalWalletBalance"])
        unrealized_pnl = float(account["totalUnrealizedProfit"])
        available_balance = float(account["availableBalance"])
        margin_balance = float(account["totalMarginBalance"])

        positions = []
        for pos in account.get("positions", []):
            amt = float(pos["positionAmt"])
            if amt != 0:
                entry_price = float(pos["entryPrice"])
                mark_price = float(pos.get("markPrice", 0))
                pnl = float(pos["unrealizedProfit"])
                leverage = int(pos.get("leverage", 1))
                side = "LONG" if amt > 0 else "SHORT"

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
                })

        if available_balance >= margin_balance * 0.5:
            fund_status, fund_color = "여유", "green"
        elif available_balance >= margin_balance * 0.2:
            fund_status, fund_color = "보통", "amber"
        else:
            fund_status, fund_color = "위험", "red"

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

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
