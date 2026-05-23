import aiohttp
import asyncio

# 봇 내부 상태 저장용 글로벌 변수들
is_initialized = False
total_assets = 0       # 가상 총자산 (원화 기준)
current_position = None # None, "LONG", "SHORT"
position_units = 0     # 보유 중인 가상 코인 수량
avg_buy_price = 0      # 현재 나의 평균 단가
entry_count = 0        # 현재 진입 차수 (0 ~ 3회, 최대 4번 진입)

# 💡 [핵심 추가] 현재 포지션 내에서 '30 이하' 또는 '70 이상' 조건이 활성화되었다가 
# 50을 안 찍고 다시 조건 구역으로 진입했는지 감시하기 위한 플래그 변수
was_outside_zone = False 

BETTING_RATIOS = [0.10, 0.20, 0.40, 0.30] # 10% -> 20% -> 40% -> 30%

async def trade_logic(seed_money):
    global is_initialized, total_assets, current_position, position_units, avg_buy_price, entry_count, was_outside_zone
    
    if not is_initialized:
        total_assets = seed_money
        current_position = None
        position_units = 0
        avg_buy_price = 0
        entry_count = 0
        was_outside_zone = False
        is_initialized = True
        print(f"🤖 RSI 파동 기준 마틴게일 봇 가동! 초기 자산: {total_assets:,}원")

    url = "https://api.upbit.com/v1/candles/minutes/1?market=KRW-BTC&count=15"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data[::-1]
                    current_price = candles[-1]['trade_price']
                    rsi = calculate_rsi(candles)
                    
                    print(f"⏰ 가격: {current_price:,}원 | RSI: {rsi:.2f} | 포지션: {current_position} (진입: {entry_count}/4) | 평단: {avg_buy_price:,}원")
                    
                    # -----------------------------------------------------------
                    # 포지션이 없는 상태 ➡️ 최초 진입 타점 (RSI 30 이하 / 70 이상)
                    # -----------------------------------------------------------
                    if current_position is None:
                        if rsi <= 30:
                            current_position = "LONG"
                            entry_count = 1
                            was_outside_zone = False # 30 이하 구역에 진입했으므로 플래그 초기화
                            invest_money = total_assets * BETTING_RATIOS[0]
                            position_units = invest_money / current_price
                            avg_buy_price = current_price
                            print(f"🟩 [롱 오픈] RSI {rsi:.2f} 1차 진입({int(BETTING_RATIOS[0]*100)}%): {invest_money:,}원 매수")
                            
                        elif rsi >= 70:
                            current_position = "SHORT"
                            entry_count = 1
                            was_outside_zone = False
                            invest_money = total_assets * BETTING_RATIOS[0]
                            position_units = invest_money / current_price
                            avg_buy_price = current_price
                            print(f"🟥 [숏 오픈] RSI {rsi:.2f} 1차 진입({int(BETTING_RATIOS[0]*100)}%): {invest_money:,}원 매도")

                    # -----------------------------------------------------------
                    # 🟩 롱 포지션 상태일 때 관리 (RSI 기준 물타기 및 익절)
                    # -----------------------------------------------------------
                    elif current_position == "LONG":
                        # 1. 청산 조건 (RSI 50 이상)
                        if rsi >= 50:
                            pnl = (current_price - avg_buy_price) * position_units
                            total_assets += pnl
                            print(f"💰 [롱 청산] RSI {rsi:.2f} 익절 완료! 손익: {pnl:+,0f}원 | 총자산: {total_assets:,}원")
                            current_position, position_units, avg_buy_price, entry_count = None, 0, 0, 0
                        
                        else:
                            # 2. RSI가 30보다 커지면 탈출 플래그를 True로 세팅 (30 이하 구역을 한 번 나갔다 왔음을 기록)
                            if rsi > 30:
                                was_outside_zone = True
                            
                            # 3. 💡 [유저 전략 핵심] 30 위로 탈출했다가(was_outside_zone==True) 50을 못 찍고 다시 30 이하로 떨어질 때 물타기!
                            if entry_count < 4 and rsi <= 30 and was_outside_zone:
                                invest_money = total_assets * BETTING_RATIOS[entry_count]
                                new_units = invest_money / current_price
                                
                                total_units = position_units + new_units
                                avg_buy_price = ((avg_buy_price * position_units) + (current_price * new_units)) / total_units
                                position_units = total_units
                                entry_count += 1
                                was_outside_zone = False # 다시 30 이하로 들어왔으니 플래그를 끄고 다음 파동을 기다림
                                print(f"💧 [롱 물타기 포착] RSI {rsi:.2f} 쌍바닥 진입! {entry_count}차 물타기({int(BETTING_RATIOS[entry_count-1]*100)}%) 진행. 새 평단가: {avg_buy_price:,.0f}원")

                    # -----------------------------------------------------------
                    # 🟥 숏 포지션 상태일 때 관리 (RSI 기준 물타기 및 익절)
                    # -----------------------------------------------------------
                    elif current_position == "SHORT":
                        # 1. 청산 조건 (RSI 50 이하)
                        if rsi <= 50:
                            pnl = (avg_buy_price - current_price) * position_units
                            total_assets += pnl
                            print(f"💰 [숏 청산] RSI {rsi:.2f} 익절 완료! 손익: {pnl:+,0f}원 | 총자산: {total_assets:,}원")
                            current_position, position_units, avg_buy_price, entry_count = None, 0, 0, 0
                        
                        else:
                            # 2. RSI가 70보다 작아지면 탈출 플래그 세팅
                            if rsi < 70:
                                was_outside_zone = True
                            
                            # 3. 💡 70 밑으로 내려갔다가 50을 못 찍고 다시 70 위로 치솟을 때 숏 물타기!
                            if entry_count < 4 and rsi >= 70 and was_outside_zone:
                                invest_money = total_assets * BETTING_RATIOS[entry_count]
                                new_units = invest_money / current_price
                                
                                total_units = position_units + new_units
                                avg_buy_price = ((avg_buy_price * position_units) + (current_price * new_units)) / total_units
                                position_units = total_units
                                entry_count += 1
                                was_outside_zone = False
                                print(f"💧 [숏 물타기 포착] RSI {rsi:.2f} 쌍봉 진입! {entry_count}차 물타기({int(BETTING_RATIOS[entry_count-1]*100)}%) 진행. 새 평단가: {avg_buy_price:,.0f}원")

                else:
                    print(f"❌ 업비트 통신 에러: {response.status}")
    except Exception as e:
        print(f"❌ 실행 중 에러: {str(e)}")

def calculate_rsi(candles):
    gains, losses = 0, 0
    for i in range(1, len(candles)):
        diff = candles[i]['trade_price'] - candles[i-1]['trade_price']
        if diff > 0: gains += diff
        else: losses += abs(diff)
    if losses == 0: return 100
    return 100 - (100 / (1 + (gains / losses)))
