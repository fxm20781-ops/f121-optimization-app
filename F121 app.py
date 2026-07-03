import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="F121 製程操作最佳化系統", layout="wide")
st.title("🏭 F121 加熱爐操作最佳化與預測系統")

# 1. 讀取數據與模型訓練（強效快取）
@st.cache_resource
def load_data_and_train_models():
    # 強迫指定 engine="openpyxl"，不論副檔名是啥，都用 Excel 格式打開
    df = pd.read_excel("data.xlsx", skiprows=[1], engine="openpyxl")
    
    # 徹底清洗所有欄位名稱：移除換行符 \n、移除所有前後重複空白
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 預設名稱
    col_dt = 'DT operation'
    col_c141 = 'C141 operation'
    col_clo = 'F121 CLO circulation flow'
    col_outlet = 'F121outlet temperature'
    col_oxy = 'F121 Oxygen content %'
    col_ng = 'F121 NG consumption'
    col_c122 = 'C122 bottom temperature'

    # 自動動態防禦機制：防止空格細微不一致
    for col in df.columns:
        if 'DT' in col and 'operation' in col: col_dt = col
        elif 'C141' in col and 'operation' in col: col_c141 = col
        elif 'CLO' in col and 'flow' in col: col_clo = col
        elif 'outlet' in col and 'temperature' in col: col_outlet = col
        elif 'Oxygen' in col: col_oxy = col
        elif 'NG' in col and 'consumption' in col: col_ng = col
        elif 'C122' in col and 'bottom' in col: col_c122 = col

    # 將所有需要用到的製程特徵欄位，強制轉換成數字並濾除異常符號（如 ***）
    target_cols = [col_dt, col_c141, col_clo, col_outlet, col_oxy, col_ng, col_c122]
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=target_cols)

    # 訓練預測模型
    features = [col_dt, col_c141, col_clo, col_outlet, col_oxy]
    X = df[features]
    y_ng = df[col_ng]
    y_c122 = df[col_c122]
    
    model_ng = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
    model_ng.fit(X, y_ng)
    
    model_c122 = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
    model_c122.fit(X, y_c122)
    
    ranges = {
        'dt_min': float(df[col_dt].min()), 'dt_max': float(df[col_dt].max()),
        'c141_min': float(df[col_c141].min()), 'c141_max': float(df[col_c141].max()),
        'flow_min': float(df[col_clo].min()), 'flow_max': float(df[col_clo].max()),
        'temp_min': float(df[col_outlet].min()), 'temp_max': float(df[col_outlet].max()),
        'oxy_min': float(df[col_oxy].min()), 'oxy_max': float(df[col_oxy].max()),
    }
    return model_ng, model_c122, ranges, features, df, target_cols

# 執行載入
try:
    with st.spinner('📊 AI 正在解析數據並建立製程預測模型...'):
        model_ng, model_c122, r, features, raw_df, target_cols = load_data_and_train_models()
    st.success('✅ 數據加載與清洗完畢，智慧優化系統已就緒。')
except Exception as e:
    st.error(f"❌ 數據初始化失敗，錯誤原因: {e}")
    st.stop()

# 2. 側邊欄輸入
st.sidebar.header("📥 輸入當前給定條件")
input_dt = st.sidebar.number_input("1. DT operation", min_value=r['dt_min'], max_value=r['dt_max'], value=0.8000, format="%.4f")
input_c141 = st.sidebar.number_input("2. C141 operation", min_value=r['c141_min'], max_value=r['c141_max'], value=1.48, format="%.2f")

# 設定預設值為歷史溫度的中間值
default_temp = float(np.clip(260.0, r['temp_min'], r['temp_max'])) # 防禦性預設值
input_outlet = st.sidebar.number_input("3. F121 outlet temperature (°C)", min_value=r['temp_min'], max_value=r['temp_max'], value=default_temp, format="%.2f")

# 3. 核心最佳化運算
if st.sidebar.button("🚀 開始計算最優操作參數", type="primary"):
    with st.spinner('🔄 正在從歷史數據中動態檢索最優節能操作點...'):
        
        # 📌 改採「歷史最鄰近實戰搜尋法」：
        # 計算歷史上每一筆紀錄跟目前設定的 [DT, C141, 出口溫度] 的相對接近程度 (歐氏距離縮放)
        # 這樣做能確保推薦出來的操作參數絕對是在工廠裡真實發生過、極度合理的數字！
        
        # 為了避免各欄位權重不均，進行簡單的範圍正規化距離計算
        dist_dt = ((raw_df[features[0]] - input_dt) / (r['dt_max'] - r['dt_min'] + 1e-5)) ** 2
        dist_c141 = ((raw_df[features[1]] - input_c141) / (r['c141_max'] - r['c141_min'] + 1e-5)) ** 2
        dist_temp = ((raw_df[features[3]] - input_outlet) / (r['temp_max'] - r['temp_min'] + 1e-5)) ** 2
        
        total_distance = np.sqrt(dist_dt + dist_c141 + dist_temp)
        
        # 篩選出歷史上最接近目前製程條件的前 10% 紀錄 (或者是最近的 50 筆)
        top_k = max(20, int(len(raw_df) * 0.05))
        closest_indices = total_distance.nsmallest(top_k).index
        candidate_df = raw_df.loc[closest_indices].copy()
        
        # 在這些極度接近當前製程條件的歷史紀錄中，找出【天然氣能耗最低】的那一筆操作
        best_row = candidate_df.loc[[candidate_df[target_cols[5]].idxmin()]]
        
        # 提取最優推薦值
        opt_flow = best_row[features[2]].values[0]
        opt_oxy = best_row[features[4]].values[0]
        
        # 透過 AI 模型來預估在此最優推薦操作下的最終能耗與副反應
        pred_input = pd.DataFrame([[input_dt, input_c141, opt_flow, input_outlet, opt_oxy]], columns=features)
        min_ng = model_ng.predict(pred_input)[0]
        predicted_c122_temp = model_c122.predict(pred_input)[0]

    # 4. 顯示結果
    st.markdown("---")
    st.header("💡 系統優化引導報告")
    st.info(f"📌 **當前給定條件基準**：DT={input_dt:.4f} | C141={input_c141:.2f} | 出口溫度={input_outlet:.2f} °C")
    
    col1, col2 = st.columns(2)
    with col1: st.metric(label="👉 建議 F121 CLO flow (歷史實戰最佳值)", value=f"{opt_flow:.2f}")
    with col2: st.metric(label="👉 建議 F121 含氧量 % (歷史實戰最佳值)", value=f"{opt_oxy:.2f} %")
        
    st.markdown("### 📈 預期效益與副反應預測")
    st.success(f"🔥 預估最低 **F121 NG consumption (能耗)**: **{min_ng:.2f}**")
    st.warning(f"🌡️ 預估此時 **C122 bottom temperature**: **{predicted_c122_temp:.2f} °C**")
else:
    st.info("👈 請在左側輸入當前的 `DT`、`C141` 與 `F121 outlet temperature`，然後點擊「開始計算最優操作參數」。")
