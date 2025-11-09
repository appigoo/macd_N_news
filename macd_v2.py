import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests  # 用於 Telegram API 請求
import time  # 用於自動刷新時間檢查

# 嘗試導入 streamlit-autorefresh 以支援自動刷新
try:
    from streamlit_autorefresh import st_autorefresh
    autorefresh_available = True
except ImportError:
    st_autorefresh = None
    autorefresh_available = False

# 計算 MACD
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# 計算 RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 計算 Stochastic
def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

# 計算 OBV
def calculate_obv(df):
    sign = np.sign(df['Close'].diff())
    obv = (sign * df['Volume']).fillna(0).cumsum()
    return obv

# 計算 MFI
def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0).rolling(window=period).sum()
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0).rolling(window=period).sum()
    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

# 計算 Bollinger Bands
def calculate_bb(df, period=20, std=2):
    sma = df['Close'].rolling(window=period).mean()
    std_dev = df['Close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

# 發送 Telegram 通知
def send_telegram_notification(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'  # 支援簡單格式化
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            st.success("Telegram 通知已發送！")
        else:
            st.error(f"Telegram 通知失敗: {response.status_code}")
    except Exception as e:
        st.error(f"發送 Telegram 通知時出錯: {e}")

# 檢測多頭分歧
def detect_bullish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_lows = pd.to_numeric(df['Low'].iloc[-3:], errors='coerce')
    hist_lows = pd.to_numeric(histogram.iloc[-3:], errors='coerce')
    diff_lows = recent_lows.diff().dropna()
    diff_hists = hist_lows.diff().dropna()
    # 確保數值比較
    lows_decreasing = all(pd.to_numeric(d, errors='coerce') <= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_lows)
    hist_decreasing = all(pd.to_numeric(d, errors='coerce') <= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hists)
    # 多頭分歧判斷是價格創新低，但指標沒有創新低
    if lows_decreasing and not hist_decreasing:
        return True
    return False

# 檢測熊頭分歧
def detect_bearish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_highs = pd.to_numeric(df['High'].iloc[-3:], errors='coerce')
    hist_highs = pd.to_numeric(histogram.iloc[-3:], errors='coerce')
    diff_highs = recent_highs.diff().dropna()
    diff_hists = hist_highs.diff().dropna()
    # 確保數值比較
    highs_increasing = all(pd.to_numeric(d, errors='coerce') >= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_highs)
    hist_increasing = all(pd.to_numeric(d, errors='coerce') >= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hists)
    # 熊頭分歧判斷是價格創新高，但指標沒有創新高
    if highs_increasing and not hist_increasing:
        return True
    return False

# 獲取數據
def get_data(ticker, period, interval):
    try:
        # 嘗試使用 Ticker.history 以避免 download 的某些錯誤
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if data.empty:
            # 後備：嘗試每日數據（適合周末）
            is_weekend = datetime.now().weekday() >= 5
            if is_weekend:
                data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        
        # 確保 OHLCV 為數值型
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=['Close'])  # 移除無效行
        
        return data
    except Exception as e:
        st.error(f"獲取數據失敗 ({ticker}): {e}")
        # 後備每日數據
        try:
            data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
            if not data.empty:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                return data
        except:
            pass
        return pd.DataFrame()

# 計算單一股票的指標和信號
def analyze_stock(ticker, period, interval, macd_fast, macd_slow, macd_signal, rsi_period, stoch_k, stoch_d, mfi_period, bb_period, bb_std):
    data = get_data(ticker, period, interval)
    if data.empty:
        return None

    required_cols = ['Close', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return None

    data = data.tail(500)  # 限制數據長度

    macd_line, signal_line, histogram = calculate_macd(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data['MACD'] = macd_line
    data['Signal'] = signal_line
    data['Histogram'] = histogram

    data['RSI'] = calculate_rsi(data, period=rsi_period)
    k, d = calculate_stochastic(data, k_period=stoch_k, d_period=stoch_d)
    data['%K'] = k
    data['%D'] = d
    data['OBV'] = calculate_obv(data)
    data['MFI'] = calculate_mfi(data, period=mfi_period)
    upper, middle, lower = calculate_bb(data, period=bb_period, std=bb_std)
    data['BB_upper'] = upper
    data['BB_middle'] = middle
    data['BB_lower'] = lower
    data = data.dropna()

    if len(data) < 10:
        return None

    latest_hist = pd.to_numeric(data['Histogram'].tail(3), errors='coerce')
    diff_hist = latest_hist.diff().dropna()
    # 確保數值比較
    hist_increasing = all(pd.to_numeric(d, errors='coerce') > 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hist) and (latest_hist.iloc[-1] < 0)
    divergence = detect_bullish_divergence(data, data['Histogram'])
    bearish_divergence = detect_bearish_divergence(data, data['Histogram'])
    rsi_latest = data['RSI'].iloc[-1]
    rsi_signal = (rsi_latest > 40) and (data['RSI'].iloc[-2] < 30) if len(data) > 1 else False
    rsi_sell_signal = (rsi_latest < 60) and (data['RSI'].iloc[-2] > 70) if len(data) > 1 else False
    stoch_cross = (data['%K'].iloc[-1] > data['%D'].iloc[-1]) and (data['%K'].iloc[-2] < 20) if len(data) > 1 else False
    stoch_sell_cross = (data['%K'].iloc[-1] < data['%D'].iloc[-1]) and (data['%K'].iloc[-2] > 80) if len(data) > 1 else False
    vol_mean = data['Volume'].rolling(10).mean().iloc[-1]
    volume_spike = (not pd.isna(vol_mean)) and (data['Volume'].iloc[-1] > vol_mean * 1.5) if len(data) > 10 else False
    volume_sell_spike = volume_spike and (data['Close'].iloc[-1] < data['Close'].iloc[-2]) if len(data) > 1 else False
    obv_up = (data['OBV'].diff().iloc[-1] > 0) if len(data) > 1 else False
    obv_down = (data['OBV'].diff().iloc[-1] < 0) if len(data) > 1 else False
    mfi_signal = (data['MFI'].iloc[-1] > 20) and (data['MFI'].iloc[-2] < 20) if len(data) > 1 else False
    mfi_sell_signal = (data['MFI'].iloc[-1] < 80) and (data['MFI'].iloc[-2] > 80) if len(data) > 1 else False
    bb_signal = data['Close'].iloc[-1] < data['BB_lower'].iloc[-1] if len(data) > 0 else False
    bb_sell_signal = data['Close'].iloc[-1] > data['BB_upper'].iloc[-1] if len(data) > 0 else False

    # 買入信號
    buy_signals = [hist_increasing, divergence, rsi_signal, stoch_cross, volume_spike, obv_up, mfi_signal, bb_signal]
    buy_score = sum(buy_signals)

    # 賣出信號（對應相反邏輯）
    hist_decreasing = all(pd.to_numeric(d, errors='coerce') < 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hist) and (latest_hist.iloc[-1] > 0)
    sell_signals = [hist_decreasing, bearish_divergence, rsi_sell_signal, stoch_sell_cross, volume_sell_spike, obv_down, mfi_sell_signal, bb_sell_signal]
    sell_score = sum(sell_signals)

    buy_suggestion = '無明顯買入信號。繼續監測。'
    if buy_score >= 3:
        buy_suggestion = '潛在買入機會：MACD Histogram 縮小，預測 MACD 可能即將從負轉正。建議關注。'
    if buy_score >= 5:
        buy_suggestion = '強烈買入信號：多指標確認，預測 MACD 即將交叉轉正。考慮進場，設止損。'

    sell_suggestion = '無明顯賣出信號。繼續持有。'
    if sell_score >= 3:
        sell_suggestion = '潛在賣出機會：MACD Histogram 擴大，預測 MACD 可能即將從正轉負。建議關注。'
    if sell_score >= 5:
        sell_suggestion = '強烈賣出信號：多指標確認，預測 MACD 即將交叉轉負。考慮出場，設止盈。'

    # 檢查是否發送 Telegram 通知
    telegram_sent_buy = False
    telegram_sent_sell = False
    if buy_score >= 5 and enable_telegram_buy and telegram_ready:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"<b>🚨 強烈買入信號！</b>\n股票: {ticker}\n時間: {now}\n收盤價: {data['Close'].iloc[-1]:.2f}\n信號強度: {buy_score}/8\n建議: {buy_suggestion}"
        send_telegram_notification(message)
        telegram_sent_buy = True

    if sell_score >= 5 and enable_telegram_sell and telegram_ready:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"<b>⚠️ 強烈賣出信號！</b>\n股票: {ticker}\n時間: {now}\n收盤價: {data['Close'].iloc[-1]:.2f}\n信號強度: {sell_score}/8\n建議: {sell_suggestion}"
        send_telegram_notification(message)
        telegram_sent_sell = True

    return {
        'ticker': ticker,
        'close': data['Close'].iloc[-1],
        'buy_score': buy_score,
        'sell_score': sell_score,
        'buy_suggestion': buy_suggestion,
        'sell_suggestion': sell_suggestion,
        'rsi': rsi_latest,
        'data': data,  # 保留數據用於詳細顯示
        'telegram_buy': telegram_sent_buy,
        'telegram_sell': telegram_sent_sell
    }

# Streamlit app 主介面
st.title('股票日內交易助手（多股票監控）')
st.write('基於 MACD、Histogram 變化、多頭分歧、RSI、Stochastic、OBV、MFI、BB 指標，自動更新。支援多股票監控。')

# Telegram 設定（整合用戶提供的 try 塊）
telegram_ready = False
try:
    # 假設 secrets.toml 已經設定
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    st.warning("Telegram 設定未完成，請在 .streamlit/secrets.toml 中添加 BOT_TOKEN 和 CHAT_ID。")

# 側邊欄輸入參數
with st.sidebar:
    st.subheader('自訂參數')
    ticker_input = st.text_input('股票代碼 (逗號分隔, 如: TSLA,AAPL,GOOGL)', value='TSLA')
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    period = st.selectbox('數據天數', ['1d', '5d', '10d'], index=1)  # 默認 5d 以避免周末 1d 問題
    interval = st.selectbox('K線間隔', ['1m', '5m', '15m', '1d'], index=1)  # 添加 1d 選項
    refresh_minutes = st.number_input('建議刷新間隔（分鐘）', value=5, min_value=1)

    # 自動刷新選項
    enable_auto_refresh = st.checkbox('啟用自動刷新', value=False)
    if enable_auto_refresh:
        auto_interval_minutes = st.selectbox('自動刷新間隔 (分鐘)', [1, 2, 3, 4, 5], index=0)
        if not autorefresh_available:
            st.warning("要使用自動刷新，請安裝 `streamlit-autorefresh`: `pip install streamlit-autorefresh`")
    else:
        auto_interval_minutes = 0

    st.subheader('指標設置')
    macd_fast = st.number_input('MACD Fast Period', value=12, min_value=1)
    macd_slow = st.number_input('MACD Slow Period', value=26, min_value=1)
    macd_signal = st.number_input('MACD Signal Period', value=9, min_value=1)
    rsi_period = st.number_input('RSI Period', value=14, min_value=1)
    stoch_k = st.number_input('Stochastic K Period', value=14, min_value=1)
    stoch_d = st.number_input('Stochastic D Period', value=3, min_value=1)
    mfi_period = st.number_input('MFI Period', value=14, min_value=1)
    bb_period = st.number_input('BB Period', value=20, min_value=1)
    bb_std = st.number_input('BB Std Dev', value=2.0, min_value=0.1, step=0.1)

    # Telegram 通知選項
    if telegram_ready:
        enable_telegram_buy = st.checkbox('啟用買入 Telegram 通知（強烈買入信號時發送）', value=False)
        enable_telegram_sell = st.checkbox('啟用賣出 Telegram 通知（強烈賣出信號時發送）', value=False)
    else:
        enable_telegram_buy = False
        enable_telegram_sell = False
        st.info("啟用 Telegram 前，請設定 secrets.toml。")

# 自動刷新邏輯（使用 streamlit-autorefresh）
if enable_auto_refresh and autorefresh_available and auto_interval_minutes > 0:
    st_autorefresh(interval=auto_interval_minutes * 60 * 1000, limit=None, key='auto_refresh')

placeholder = st.empty()

# 選擇顯示詳細的股票
selected_ticker = st.selectbox('選擇顯示詳細圖表的股票', tickers) if tickers else None

def refresh_data():
    if not tickers:
        with placeholder:
            st.error('請輸入至少一個股票代碼。')
        return

    results = []
    for ticker in tickers:
        result = analyze_stock(ticker, period, interval, macd_fast, macd_slow, macd_signal, rsi_period, stoch_k, stoch_d, mfi_period, bb_period, bb_std)
        if result:
            results.append(result)

    if not results:
        with placeholder:
            st.error('無法獲取任何股票數據，請檢查代碼或調整參數。')
        return

    # 顯示多股票摘要表格
    summary_df = pd.DataFrame([
        {
            '股票': r['ticker'],
            '收盤價': f"{r['close']:.2f}",
            '買入分數': r['buy_score'],
            '賣出分數': r['sell_score'],
            'RSI': f"{r['rsi']:.2f}",
            '買入建議': r['buy_suggestion'][:50] + '...' if len(r['buy_suggestion']) > 50 else r['buy_suggestion'],
            '賣出建議': r['sell_suggestion'][:50] + '...' if len(r['sell_suggestion']) > 50 else r['sell_suggestion']
        }
        for r in results
    ])

    with placeholder:
        st.subheader('多股票監控摘要')
        st.dataframe(summary_df, use_container_width=True)

        # 高亮強烈信號
        strong_buy = [r for r in results if r['buy_score'] >= 5]
        strong_sell = [r for r in results if r['sell_score'] >= 5]
        if strong_buy:
            st.warning(f"強烈買入信號股票: {', '.join([r['ticker'] for r in strong_buy])}")
        if strong_sell:
            st.error(f"強烈賣出信號股票: {', '.join([r['ticker'] for r in strong_sell])}")

        if selected_ticker:
            # 顯示選中股票的詳細資訊
            selected_result = next((r for r in results if r['ticker'] == selected_ticker), None)
            if selected_result:
                data = selected_result['data']
                hist_increasing = all(pd.to_numeric(d, errors='coerce') > 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in pd.to_numeric(data['Histogram'].tail(3), errors='coerce').diff().dropna()) and (pd.to_numeric(data['Histogram'].tail(3), errors='coerce').iloc[-1] < 0)
                hist_decreasing = all(pd.to_numeric(d, errors='coerce') < 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in pd.to_numeric(data['Histogram'].tail(3), errors='coerce').diff().dropna()) and (pd.to_numeric(data['Histogram'].tail(3), errors='coerce').iloc[-1] > 0)
                divergence = detect_bullish_divergence(data, data['Histogram'])
                bearish_divergence = detect_bearish_divergence(data, data['Histogram'])
                rsi_latest = data['RSI'].iloc[-1]
                rsi_signal = (rsi_latest > 40) and (data['RSI'].iloc[-2] < 30) if len(data) > 1 else False
                rsi_sell_signal = (rsi_latest < 60) and (data['RSI'].iloc[-2] > 70) if len(data) > 1 else False
                stoch_cross = (data['%K'].iloc[-1] > data['%D'].iloc[-1]) and (data['%K'].iloc[-2] < 20) if len(data) > 1 else False
                stoch_sell_cross = (data['%K'].iloc[-1] < data['%D'].iloc[-1]) and (data['%K'].iloc[-2] > 80) if len(data) > 1 else False
                vol_mean = data['Volume'].rolling(10).mean().iloc[-1]
                volume_spike = (not pd.isna(vol_mean)) and (data['Volume'].iloc[-1] > vol_mean * 1.5) if len(data) > 10 else False
                volume_sell_spike = volume_spike and (data['Close'].iloc[-1] < data['Close'].iloc[-2]) if len(data) > 1 else False
                obv_up = (data['OBV'].diff().iloc[-1] > 0) if len(data) > 1 else False
                obv_down = (data['OBV'].diff().iloc[-1] < 0) if len(data) > 1 else False
                mfi_signal = (data['MFI'].iloc[-1] > 20) and (data['MFI'].iloc[-2] < 20) if len(data) > 1 else False
                mfi_sell_signal = (data['MFI'].iloc[-1] < 80) and (data['MFI'].iloc[-2] > 80) if len(data) > 1 else False
                bb_signal = data['Close'].iloc[-1] < data['BB_lower'].iloc[-1] if len(data) > 0 else False
                bb_sell_signal = data['Close'].iloc[-1] > data['BB_upper'].iloc[-1] if len(data) > 0 else False

                st.subheader(f'{selected_ticker} 詳細數據和指標')
                st.metric("最新收盤價", f"{data['Close'].iloc[-1]:.2f}")
                st.write(f'MACD Histogram: {data["Histogram"].iloc[-1]:.4f} (買入縮小: {"是" if hist_increasing else "否"}, 賣出擴大: {"是" if hist_decreasing else "否"})')
                st.write(f'多頭分歧: {"檢測到" if divergence else "無"} | 熊頭分歧: {"檢測到" if bearish_divergence else "無"}')
                st.write(f'RSI: {rsi_latest:.2f} (買入信號: {"是" if rsi_signal else "否"}, 賣出信號: {"是" if rsi_sell_signal else "否"})')
                st.write(f'Stochastic %K/%D: {data["%K"].iloc[-1]:.2f} / {data["%D"].iloc[-1]:.2f} (買入交叉: {"是" if stoch_cross else "否"}, 賣出交叉: {"是" if stoch_sell_cross else "否"})')
                st.write(f'OBV: {data["OBV"].iloc[-1]:,.0f} (上漲: {"是" if obv_up else "否"}, 下跌: {"是" if obv_down else "否"})')
                st.write(f'MFI: {data["MFI"].iloc[-1]:.2f} (買入信號: {"是" if mfi_signal else "否"}, 賣出信號: {"是" if mfi_sell_signal else "否"})')
                st.write(f'Bollinger Bands: Close vs Lower/Upper: {data["Close"].iloc[-1]:.2f} vs {data["BB_lower"].iloc[-1]:.2f} / {data["BB_upper"].iloc[-1]:.2f} (買入觸底: {"是" if bb_signal else "否"}, 賣出觸頂: {"是" if bb_sell_signal else "否"})')
                st.write(f'成交量尖峰 (買入): {"是" if volume_spike else "否"} | (賣出): {"是" if volume_sell_spike else "否"}')

                st.subheader('買入交易建議')
                st.write(selected_result['buy_suggestion'])
                st.write(f'買入信號強度: {selected_result["buy_score"]}/8')

                st.subheader('賣出交易建議')
                st.write(selected_result['sell_suggestion'])
                st.write(f'賣出信號強度: {selected_result["sell_score"]}/8')

                st.subheader('最近 10 根 K 線數據')
                st.dataframe(data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']])

                # 成交量走勢圖
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader('價格走勢')
                    st.line_chart(data['Close'].tail(50))
                with col2:
                    st.subheader('MACD Histogram')
                    st.line_chart(data['Histogram'].tail(50))
                with col3:
                    st.subheader('成交量')
                    st.bar_chart(data['Volume'].tail(50))

# 初始載入數據
refresh_data()

# 手動刷新按鈕（側邊欄參數變化時自動 reruns）
st.sidebar.markdown("---")
if st.sidebar.button('立即刷新數據'):
    st.rerun()

st.sidebar.info(f'建議每 {refresh_minutes} 分鐘手動刷新一次，以獲取最新數據。周末將自動切換至每日數據。')
if enable_auto_refresh:
    if autorefresh_available:
        st.sidebar.success(f'自動刷新已啟用，每 {auto_interval_minutes} 分鐘一次。')
    else:
        st.sidebar.error('自動刷新不可用，請安裝 streamlit-autorefresh。')
