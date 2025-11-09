import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests  # 用於 Telegram API 請求和新新聞 API

# 新增：翻譯庫
try:
    from googletrans import Translator
    translator = Translator()
    translation_available = True
except ImportError:
    translator = None
    translation_available = False

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

# 發送 Telegram 通知（添加防重發邏輯）
def send_telegram_notification(message, last_sent_time=None):
    if last_sent_time and (datetime.now() - last_sent_time).seconds < 60:  # 防 1 分內重發
        st.info("通知已於最近發送，跳過。")
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            st.success("Telegram 通知已發送！")
            return True
        else:
            st.error(f"Telegram 通知失敗: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"發送 Telegram 通知時出錯: {e}")
        return False

# 檢測多頭分歧（修復 NaN/空 diff 處理）
def detect_bullish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_lows = pd.to_numeric(df['Low'].iloc[-3:], errors='coerce').dropna()
    hist_lows = pd.to_numeric(histogram.iloc[-3:], errors='coerce').dropna()
    if len(recent_lows) < 2 or len(hist_lows) < 2:  # 新增：確保足夠數據
        return False
    diff_lows = recent_lows.diff().dropna()
    diff_hists = hist_lows.diff().dropna()
    if len(diff_lows) < 1 or len(diff_hists) < 1:  # 新增：空 diff 檢查
        return False
    lows_decreasing = all(d <= 0 and not pd.isna(d) for d in diff_lows)
    hist_decreasing = all(d <= 0 and not pd.isna(d) for d in diff_hists)
    return lows_decreasing and not hist_decreasing

# 檢測熊頭分歧（同上修復）
def detect_bearish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_highs = pd.to_numeric(df['High'].iloc[-3:], errors='coerce').dropna()
    hist_highs = pd.to_numeric(histogram.iloc[-3:], errors='coerce').dropna()
    if len(recent_highs) < 2 or len(hist_highs) < 2:
        return False
    diff_highs = recent_highs.diff().dropna()
    diff_hists = hist_highs.diff().dropna()
    if len(diff_highs) < 1 or len(diff_hists) < 1:
        return False
    highs_increasing = all(d >= 0 and not pd.isna(d) for d in diff_highs)
    hist_increasing = all(d >= 0 and not pd.isna(d) for d in diff_hists)
    return highs_increasing and not hist_increasing

# 獲取數據（添加快取）
@st.cache_data(ttl=300)  # 新增：5 分快取
def get_data(ticker, period, interval):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, prepost=True)  # 新增 prepost=True (2025 推薦)
        if data.empty:
            is_weekend = datetime.now().weekday() >= 5
            if is_weekend:
                data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False, prepost=True)
        if data.empty:
            return pd.DataFrame()
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=['Close'])
        
        return data
    except Exception as e:
        st.error(f"獲取數據失敗 ({ticker}): {e}")
        try:
            data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False, prepost=True)
            if not data.empty:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                return data
        except:
            pass
        return pd.DataFrame()

# 獲取即時新聞（添加語言選項）
def get_news(ticker, api_key, language='en'):  # 新增語言參數
    if not api_key:
        return []
    try:
        url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&sortBy=publishedAt&pageSize=5&language={language}'
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return articles
        else:
            st.error(f"新聞 API 請求失敗: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"獲取新聞失敗: {e}")
        return []

# 新增：翻譯函數
def translate_to_chinese(text):
    if not translation_available or not text:
        return text
    try:
        if translator.detect(text).lang == 'zh':
            return text  # 已為中文，跳過
        translated = translator.translate(text, dest='zh-cn').text
        return translated
    except Exception as e:
        st.warning(f"翻譯失敗 ({text[:50]}...): {e}")
        return text  # 回退原文字

# 計算單一股票的指標和信號（修復 NaN 檢查，重複計算移出）
def analyze_stock(ticker, period, interval, macd_fast, macd_slow, macd_signal, rsi_period, stoch_k, stoch_d, mfi_period, bb_period, bb_std, news_api_key, language='en'):
    data = get_data(ticker, period, interval)
    if data.empty:
        return None

    required_cols = ['Close', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return None

    data = data.tail(500)

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

    # 修復：安全計算 hist diff
    latest_hist = pd.to_numeric(data['Histogram'].tail(3), errors='coerce').dropna()
    if len(latest_hist) < 2:
        hist_increasing = hist_decreasing = False
    else:
        diff_hist = latest_hist.diff().dropna()
        hist_increasing = (len(diff_hist) >= 1) and all(d > 0 and not pd.isna(d) for d in diff_hist) and (latest_hist.iloc[-1] < 0 and not pd.isna(latest_hist.iloc[-1]))
        hist_decreasing = (len(diff_hist) >= 1) and all(d < 0 and not pd.isna(d) for d in diff_hist) and (latest_hist.iloc[-1] > 0 and not pd.isna(latest_hist.iloc[-1]))

    divergence = detect_bullish_divergence(data, data['Histogram'])
    bearish_divergence = detect_bearish_divergence(data, data['Histogram'])
    rsi_latest = data['RSI'].iloc[-1]
    rsi_signal = (not pd.isna(rsi_latest) and rsi_latest > 40) and (len(data) > 1 and not pd.isna(data['RSI'].iloc[-2]) and data['RSI'].iloc[-2] < 30)
    rsi_sell_signal = (not pd.isna(rsi_latest) and rsi_latest < 60) and (len(data) > 1 and not pd.isna(data['RSI'].iloc[-2]) and data['RSI'].iloc[-2] > 70)
    stoch_cross = (len(data) > 1 and not pd.isna(data['%K'].iloc[-1]) and not pd.isna(data['%D'].iloc[-1]) and data['%K'].iloc[-1] > data['%D'].iloc[-1]) and (not pd.isna(data['%K'].iloc[-2]) and data['%K'].iloc[-2] < 20)
    stoch_sell_cross = (len(data) > 1 and not pd.isna(data['%K'].iloc[-1]) and not pd.isna(data['%D'].iloc[-1]) and data['%K'].iloc[-1] < data['%D'].iloc[-1]) and (not pd.isna(data['%K'].iloc[-2]) and data['%K'].iloc[-2] > 80)
    vol_mean = data['Volume'].rolling(10).mean().iloc[-1]
    volume_spike = (len(data) > 10 and not pd.isna(vol_mean) and not pd.isna(data['Volume'].iloc[-1]) and data['Volume'].iloc[-1] > vol_mean * 1.5)
    volume_sell_spike = volume_spike and (len(data) > 1 and data['Close'].iloc[-1] < data['Close'].iloc[-2])
    obv_up = (len(data) > 1 and not pd.isna(data['OBV'].diff().iloc[-1]) and data['OBV'].diff().iloc[-1] > 0)
    obv_down = (len(data) > 1 and not pd.isna(data['OBV'].diff().iloc[-1]) and data['OBV'].diff().iloc[-1] < 0)
    mfi_signal = (len(data) > 1 and not pd.isna(data['MFI'].iloc[-1]) and not pd.isna(data['MFI'].iloc[-2]) and data['MFI'].iloc[-1] > 20 and data['MFI'].iloc[-2] < 20)
    mfi_sell_signal = (len(data) > 1 and not pd.isna(data['MFI'].iloc[-1]) and not pd.isna(data['MFI'].iloc[-2]) and data['MFI'].iloc[-1] < 80 and data['MFI'].iloc[-2] > 80)
    bb_signal = (len(data) > 0 and not pd.isna(data['Close'].iloc[-1]) and not pd.isna(data['BB_lower'].iloc[-1]) and data['Close'].iloc[-1] < data['BB_lower'].iloc[-1])
    bb_sell_signal = (len(data) > 0 and not pd.isna(data['Close'].iloc[-1]) and not pd.isna(data['BB_upper'].iloc[-1]) and data['Close'].iloc[-1] > data['BB_upper'].iloc[-1])

    # 買入信號
    buy_signals = [hist_increasing, divergence, rsi_signal, stoch_cross, volume_spike, obv_up, mfi_signal, bb_signal]
    buy_score = sum(buy_signals)

    # 賣出信號
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

    # Telegram 通知（添加時間戳防重）
    telegram_sent_buy = False
    telegram_sent_sell = False
    last_buy_time = st.session_state.get('last_buy_time', {}).get(ticker)
    last_sell_time = st.session_state.get('last_sell_time', {}).get(ticker)
    if buy_score >= 5 and enable_telegram_buy and telegram_ready:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"<b>🚨 強烈買入信號！</b>\n股票: {ticker}\n時間: {now}\n收盤價: {data['Close'].iloc[-1]:.2f}\n信號強度: {buy_score}/8\n建議: {buy_suggestion}"
        if send_telegram_notification(message, last_buy_time):
            st.session_state.setdefault('last_buy_time', {})[ticker] = datetime.now()
            telegram_sent_buy = True

    if sell_score >= 5 and enable_telegram_sell and telegram_ready:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"<b>⚠️ 強烈賣出信號！</b>\n股票: {ticker}\n時間: {now}\n收盤價: {data['Close'].iloc[-1]:.2f}\n信號強度: {sell_score}/8\n建議: {sell_suggestion}"
        if send_telegram_notification(message, last_sell_time):
            st.session_state.setdefault('last_sell_time', {})[ticker] = datetime.now()
            telegram_sent_sell = True

    # 獲取新聞
    news = get_news(ticker, news_api_key, language)

    # 新增：返回所有信號細節，避免重複計算
    return {
        'ticker': ticker,
        'close': data['Close'].iloc[-1],
        'buy_score': buy_score,
        'sell_score': sell_score,
        'buy_suggestion': buy_suggestion,
        'sell_suggestion': sell_suggestion,
        'rsi': rsi_latest,
        'data': data,
        'news': news,
        'telegram_buy': telegram_sent_buy,
        'telegram_sell': telegram_sent_sell,
        'signals': {  # 新增：所有信號 dict
            'hist_increasing': hist_increasing,
            'hist_decreasing': hist_decreasing,
            'divergence': divergence,
            'bearish_divergence': bearish_divergence,
            'rsi_signal': rsi_signal,
            'rsi_sell_signal': rsi_sell_signal,
            'stoch_cross': stoch_cross,
            'stoch_sell_cross': stoch_sell_cross,
            'volume_spike': volume_spike,
            'volume_sell_spike': volume_sell_spike,
            'obv_up': obv_up,
            'obv_down': obv_down,
            'mfi_signal': mfi_signal,
            'mfi_sell_signal': mfi_sell_signal,
            'bb_signal': bb_signal,
            'bb_sell_signal': bb_sell_signal
        }
    }

# Streamlit app 主介面
st.title('股票日內交易助手（多股票監控 + 即時新聞）')
st.write('基於 MACD、Histogram 變化、多頭分歧、RSI、Stochastic、OBV、MFI、BB 指標，自動更新。支援多股票監控及即時新聞饋送。')

# Telegram 設定
telegram_ready = False
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    st.warning("Telegram 設定未完成，請在 .streamlit/secrets.toml 中添加 BOT_TOKEN 和 CHAT_ID。")

# NewsAPI 設定
news_ready = False
news_api_key = None
try:
    news_api_key = st.secrets["newsapi"]["API_KEY"]
    news_ready = True
except:
    st.warning("NewsAPI 設定未完成，請在 .streamlit/secrets.toml 中添加 newsapi 區段和 API_KEY。")

# 側邊欄輸入參數
with st.sidebar:
    st.subheader('自訂參數')
    ticker_input = st.text_input('股票代碼 (逗號分隔, 如: TSLA,AAPL,GOOGL)', value='TSLA')
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    period = st.selectbox('數據天數', ['1d', '5d', '10d'], index=1)
    interval = st.selectbox('K線間隔', ['1m', '5m', '15m', '1d'], index=1)
    refresh_minutes = st.number_input('建議刷新間隔（分鐘）', value=5, min_value=1)

    # 新聞 API 設定（添加語言選項）
    st.subheader('新聞設定')
    news_language = st.selectbox('新聞語言', ['en', 'zh'], index=0)  # 新增
    enable_translation = st.checkbox('啟用新聞自動翻譯成中文', value=True)  # 新增：翻譯開關
    if not translation_available:
        st.warning("要使用翻譯，請安裝 `googletrans`: `pip install googletrans==4.0.0-rc1`")
    if not news_ready:
        st.info("**如何設定 NewsAPI 金鑰：**\n\n在 `.streamlit/secrets.toml` 檔案中添加以下內容：\n\n```toml\n[newsapi]\nAPI_KEY = \"your_newsapi_key_here\"\n```\n\n獲取金鑰：https://newsapi.org/")

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

# 自動刷新邏輯
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
        result = analyze_stock(ticker, period, interval, macd_fast, macd_slow, macd_signal, rsi_period, stoch_k, stoch_d, mfi_period, bb_period, bb_std, news_api_key, news_language)
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
            # 顯示選中股票的詳細資訊（使用返回的 signals，避免重算）
            selected_result = next((r for r in results if r['ticker'] == selected_ticker), None)
            if selected_result:
                data = selected_result['data']
                signals = selected_result['signals']
                hist_increasing = signals['hist_increasing']
                hist_decreasing = signals['hist_decreasing']
                divergence = signals['divergence']
                bearish_divergence = signals['bearish_divergence']
                rsi_latest = selected_result['rsi']
                rsi_signal = signals['rsi_signal']
                rsi_sell_signal = signals['rsi_sell_signal']
                stoch_cross = signals['stoch_cross']
                stoch_sell_cross = signals['stoch_sell_cross']
                volume_spike = signals['volume_spike']
                volume_sell_spike = signals['volume_sell_spike']
                obv_up = signals['obv_up']
                obv_down = signals['obv_down']
                mfi_signal = signals['mfi_signal']
                mfi_sell_signal = signals['mfi_sell_signal']
                bb_signal = signals['bb_signal']
                bb_sell_signal = signals['bb_sell_signal']

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

                # 新增：用 tabs 分離圖表和新聞，確保同時可見
                tab1, tab2 = st.tabs(["📈 走勢圖表", "📰 即時新聞"])

                with tab1:
                    # 新增：載入 spinner 避免閃爍
                    with st.spinner('載入圖表...'):
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

                with tab2:
                    # 新增：載入 spinner 和佔位符
                    spinner_text = '載入新聞...' if not enable_translation else '翻譯中...'
                    with st.spinner(spinner_text):
                        news = selected_result['news']
                        if news:
                            st.subheader(f'{selected_ticker} 最新新聞 (前 5 則)')
                            for i, article in enumerate(news, 1):
                                # 新增：自動翻譯
                                title_zh = translate_to_chinese(article['title']) if enable_translation else article['title']
                                desc_zh = translate_to_chinese(article['description'] or '無摘要') if enable_translation else (article['description'] or '無摘要')
                                with st.expander(f"{i}. {title_zh} - {article['publishedAt'][:19]}"):
                                    st.write(desc_zh)
                                    if article['url']:
                                        st.markdown(f"[閱讀全文]({article['url']})")
                                    st.caption(f"來源: {article['source']['name']}")
                        else:
                            if news_ready:
                                st.info("🔍 無相關新聞數據。嘗試調整股票代碼或稍後刷新。")
                                st.caption("提示：NewsAPI 可能需時間更新，或試用其他 ticker 如 AAPL。")
                            else:
                                st.warning("⚠️ 無新聞數據，請檢查 NewsAPI 金鑰設定。")
                                st.info("**快速設定步驟：**\n1. 註冊 https://newsapi.org\n2. 在 `.streamlit/secrets.toml` 添加 [newsapi] API_KEY\n3. 重新啟動 app。")

# 初始載入數據
refresh_data()

# 手動刷新按鈕
st.sidebar.markdown("---")
if st.sidebar.button('立即刷新數據'):
    st.rerun()

st.sidebar.info(f'建議每 {refresh_minutes} 分鐘手動刷新一次，以獲取最新數據。周末將自動切換至每日數據。')
if enable_auto_refresh:
    if autorefresh_available:
        st.sidebar.success(f'自動刷新已啟用，每 {auto_interval_minutes} 分鐘一次。')
    else:
        st.sidebar.error('自動刷新不可用，請安裝 streamlit-autorefresh。')
