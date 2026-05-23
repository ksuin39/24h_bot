import aiohttp
import asyncio

is_initialized = False
total_assets = 0       
current_position = None 
position_units = 0     
avg_buy_price = 0      
entry_count = 0        
was_outside_zone = False 

BETTING_RATIOS = [0.10, 0.20, 0.40, 0.30] 
WATERING_THRESHOLD = 0.015 

# 💡 leverage 인자 추가 수혈!
async def trade_logic(seed_money, target_coin, currency, leverage):
    global is_initialized, total_assets, current_position, position_units, avg_buy_price, entry_count, was_outside_zone
    
    if not is_initialized:
        total_assets = seed_money
        current_position = None
        position_units = 0
        avg_buy_price = 0
        entry_count = 0
        was_outside_zone = False
        is_initialized = True
        print(f"🤖 레버리지 엔진 탑재 완료 -> 자산: {total_assets:,}{currency} | 배율: {leverage}배")

    if currency == "KRW":
        url = f"https://api.upbit.com/v1/candles/minutes/1?market=KRW-{target_coin}&count=15"
    else:
        url = f"https://api.binance.com/api/v3/klines?symbol={target_coin}USDT&interval=1m&limit=15"
        
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if currency == "KRW":
                        candles = data[::-1]
                        current_price = candles[-1]['trade_price']
                        rsi = calculate_rsi(candles, is_upbit=True)
                    else:
                        current_price = float(data[-1][4]) 
                        rsi = calculate_rsi(data, is_upbit=False)
                    
                    unit_symbol = "원" if currency == "KRW" else "$"
                    print(f"⏰ {target_coin} 가격: {current_price:,}{unit_symbol} | RSI: {rsi:.2f} | 레버리지: {leverage}x")
                    
                    # -----------------------------------------------------------
                    # 1차 포지션 오픈 (레버리지 뻥튀기 반영)
                    # -----------------------------------------------------------
                    if current_position is None:
                        if rsi <= 30:
                            current_position = "LONG"
                            entry_count = 1
                            was_outside_zone = False
                            invest_money = total_assets * BETTING_RATIOS[0]
                            # 💡 실제 투자금에 레버리지를 곱해 계약 수량을 늘립니다!
                            position_units = (invest_money * leverage) / current_price
                            avg_buy_price = current_price
                            print(f"🟩 [롱 {leverage}x] RSI {rsi:.2f} 진입: {invest_money:,}{currency} 사용 (실제 {invest_money*leverage:,} 규모)")
                            
                        elif rsi >= 70:
                            current_position = "SHORT"
                            entry_count = 1
                            was_outside_zone = False
                            invest_money = total_assets * BETTING_RATIOS[0]
                            position_units = (invest_money * leverage) / current_price
                            avg_buy_price = current_price
                            print(f"🟥 [숏 {leverage}x] RSI {rsi:.2f} 진입: {invest_money:,}{currency} 사용 (실제 {invest_money*leverage:,} 규모)")

                    # -----------------------------------------------------------
                    # 🟩 롱 포지션 파동 운용 (수익/손실 연산 자동 뻥튀기됨)
                    # -----------------------------------------------------------
                    elif current_position == "LONG":
                        if rsi >= 50:
                            # 💡 이미 수량(position_units)에 레버리지가 곱해져 있으므로 손익도 배수로 계산됩니다.
                            pnl = (current_price - avg_buy_price) * position_units
                            total_assets += pnl
                            print(f"💰 [롱 청산] 손익: {pnl:+,0f}{currency} | 정산 후 총자산: {total_assets:,}{currency}")
                            current_position, position_units, avg_buy_price, entry_count = None, 0, 0, 0
                        else:
                            if rsi > 30: was_outside_zone = True
                            if entry_count < 4 and rsi <= 30 and was_outside_zone:
                                invest_money = total_assets * BETTING_RATIOS[entry_count]
                                # 💡 물탈 때도 설정된 레버리지 배율만큼 물 수량 뻥튀기
                                new_units = (invest_money * leverage) / current_price
                                total_units = position_units + new_units
                                avg_buy_price = ((avg_buy_price * position_units) + (current_price * new_units)) / total_units
                                position_units = total_units
                                entry_count += 1
                                was_outside_zone = False
                                print(f"💧 [롱 물타기] {entry_count}차 물타기 진행 (평단가: {avg_buy_price:,.0f})")

                    # -----------------------------------------------------------
                    # 🟥 숏 포지션 파동 운용
                    # -----------------------------------------------------------
                    elif current_position == "SHORT":
                        if rsi <= 50:
                            pnl = (avg_buy_price - current_price) * position_units
                            total_assets += pnl
                            print(f"💰 [숏 청산] 손익: {pnl:+,0f}{currency} | 정산 후 총자산: {total_assets:,}{currency}")
                            current_position, position_units, avg_buy_price, entry_count = None, 0, 0, 0
                        else:
                            if rsi < 70: was_outside_zone = True
                            if entry_count < 4 and rsi >= 70 and was_outside_zone:
                                invest_money = total_assets * BETTING_RATIOS[entry_count]
                                new_units = (invest_money * leverage) / current_price
                                total_units = position_units + new_units
                                avg_buy_price = ((avg_buy_price * position_units) + (current_price * new_units)) / total_units
                                position_units = total_units
                                entry_count += 1
                                was_outside_zone = False
                                print(f"💧 [숏 물타기] {entry_count}차 물타기 진행 (평단가: {avg_buy_price:,.0f})")
                else:
                    print(f"❌ 데이터 에러: {response.status}")
    except Exception as e:
        print(f"❌ 봇 내부 에러: {str(e)}")

def calculate_rsi(candles, is_upbit=True):
    gains, losses = 0, 0
    prices = [c['trade_price'] for c in candles] if is_upbit else [float(c[4]) for c in candles]
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if diff > 0: gains += diff
        else: losses += abs(diff)
    if losses == 0: return 100
    return 100 - (100 / (1 + (gains / losses)))
