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
    res = requests.get(url, headers=headers, timeout=15)
    return res.json()

def fetch_futures_data(api_key, secret_key):
    # 최대 3번 재시도
    last_error = ""
    for attempt in range(3):
        try:
            balances = api_get("/fapi/v3/balance", api_key, secret_key)
            if isinstance(balances, dict) and balances.get("code"):
                last_error = balances.get("msg", "API 오류")
                time.sleep(1)
                continue

            account = api_get("/fapi/v3/account", api_key, secret_key)
            if isinstance(account, dict) and account.get("code"):
                last_error = account.get("msg", "API 오류")
                time.sleep(1)
                continue

            usdt = next((b for b in balances if b["asset"] == "USDT"), None)
            total_balance     = float(usdt["balance"])           if usdt else 0
            available_balance = float(usdt["availableBalance"])  if usdt else 0
            unrealized_pnl    = float(usdt.get("crossUnPnl", 0)) if usdt else 0
            margin_balance    = float(usdt.get("marginBalance", total_balance)) if usdt else 0

            positions = []
            raw_positions = account.get("positions", [])
            if not isinstance(raw_positions, list):
                raw_positions = []

            for pos in raw_positions:
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    entry_price = float(pos.get("entryPrice", 0))
                    mark_price  = float(pos.get("markPrice", 0))
                    pnl         = float(pos.get("unrealizedProfit", 0))
                    leverage    = int(pos.get("leverage", 1))
                    side        = "LONG" if amt > 0 else "SHORT"

                    if entry_price > 0:
                        pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * leverage
                        if side == "SHORT":
                            pnl_pct = -pnl_pct
                    else:
                        pnl_pct = 0

                    positions.append({
                        "symbol":         pos["symbol"],
                        "side":           side,
                        "amount":         abs(amt),
                        "entry_price":    entry_price,
                        "mark_price":     mark_price,
                        "unrealized_pnl": pnl,
                        "pnl_pct":        round(pnl_pct, 2),
                        "leverage":       leverage,
                    })

            if margin_balance > 0:
                if available_balance >= margin_balance * 0.5:
                    fund_status, fund_color = "여유", "green"
                elif available_balance >= margin_balance * 0.2:
                    fund_status, fund_color = "보통", "amber"
                else:
                    fund_status, fund_color = "위험", "red"
            else:
                fund_status, fund_color = "여유", "green"

            return {
                "success": True,
                "account": {
                    "total_balance":     round(total_balance, 2),
                    "margin_balance":    round(margin_balance, 2),
                    "available_balance": round(available_balance, 2),
                    "unrealized_pnl":    round(unrealized_pnl, 2),
                    "fund_status":       fund_status,
                    "fund_color":        fund_color,
                },
                "positions": positions,
                "fetched_at": int(time.time()),
            }

        except Exception as e:
            last_error = str(e)
            time.sleep(1)
            continue

    return {"success": False, "error": last_error}

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
        api_key    = data.get("api_key", "").strip()
        secret_key = data.get("secret_key", "").strip()

        if not api_key or not secret_key:
            return jsonify({"success": False, "error": "API 키와 시크릿 키를 모두 입력해주세요."})

        result = fetch_futures_data(api_key, secret_key)
        if not result["success"]:
            return jsonify({"success": False, "error": result["error"]})

        session["api_key"]    = api_key
        session["secret_key"] = secret_key
        session["cache"]      = result
        return jsonify({"success": True})

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/api/futures")
def futures_data():
    if not is_logged_in():
        return jsonify({"success": False, "error": "로그인이 필요합니다."}), 401

    api_key    = session["api_key"]
    secret_key = session["secret_key"]

    # 캐시가 있고 30초 이내면 캐시 반환
    cache = session.get("cache")
    now   = int(time.time())
    if cache and (now - cache.get("fetched_at", 0)) < 30:
        return jsonify(cache)

    # 새로 fetch
    result = fetch_futures_data(api_key, secret_key)

    # 실패해도 캐시가 있으면 캐시로 응답
    if not result["success"]:
        if cache:
            cache["cache_warning"] = f"갱신 실패 (캐시 사용 중): {result['error']}"
            return jsonify(cache)
        return jsonify(result)

    session["cache"] = result
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
