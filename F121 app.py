import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import os

# --- 1. 網頁標題與設定 ---
st.set_page_config(page_title="F121 節能與製程預測系統", layout="wide")
st.title("🔥 F121 天然氣最佳化操作與 C122 溫度預測系統")

# --- 2. 讀取 Excel 真實資料與訓練雙模型 ---
@st.cache_resource
def train_models_with_real_data():
    excel_filename = "F121_Data.xlsx" # 👈 直接讀取您的 Excel 檔案
    
    if not os.path.exists(excel_filename):
        st.error(f"❌ 找不到資料檔 {excel_filename}，請確認是否有上傳到 GitHub，且檔名大小寫一致。")
        st.stop()
        
    # 讀取 Excel：自動讀取第一個分頁，並跳過第 1 行的 Tag 代號 (TR122-11 等)
    df = pd.read_excel(excel_filename, skiprows=[1]) 
    
    # 清理欄位名稱（移除換行符號與前後空格）
    df.columns = df.columns.astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 定義對應的真實欄位名稱
    X_cols = ['DT operation', 'C141 operation', 'F121 CLO circulation flow', 'F121outlet temperature', 'F121 Oxygen content %']
    y_ng_col = 'F121 NG consumption'
    y_c122_col = 'C122 bottom temperature'
    
    # 移除非數值或缺失值的資料
    all_cols = X_cols + [y_ng_col, y_c122_col]
    df_clean = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) == 0:
        st.error("❌ Excel 數據解析後為空，請檢查欄位名稱是否正確。")
        st.stop()

    X = df_clean[X_cols]
    y_ng = df_clean[y_ng_col]
    y_c122 = df_clean[y_c122_col]
    
    # 獲取各變數的真實上下限
    bounds_dict = {col: (float(X[col].min()), float(X[col].max())) for col in X_cols}
    
    # 訓練模型
    model_ng = LGBMRegressor(random_state=42)
    model_ng.fit(X, y_ng)
    
    model_c122 = LGBMRegressor(random_state=42)
    model_c122.fit(X, y_c122)
    
    return model_ng, model_c122, bounds_dict

with st.spinner("🚀 正在讀取 Excel 數據並訓練 AI 模型..."):
    model_ng, model_c122, bounds = train_models_with_real_data()

# --- 3. 側邊欄：不可控變數輸入 ---
st.sidebar.header("📋 當前不可控排程設定")
input_dt = st.sidebar.slider("DT operation (稼動率)", min_value=bounds['DT operation'][0], max_value=bounds['DT operation'][1], value=(bounds['DT operation'][0]+bounds['DT operation'][1])/2, step=0.01)
input_c141 = st.sidebar.slider("C141 operation (稼動率)", min_value=bounds['C141 operation'][0], max_value=bounds['C141 operation'][1], value=(bounds['C141 operation'][0]+bounds['C141 operation'][1])/2, step=0.01)

# --- 4. 優化演算法核心 ---
def objective_func(controllable_vars):
    features = np.array([[input_dt, input_c141, controllable_vars[0], controllable_vars[1], controllable_vars[2]]])
    return model_ng.predict(features)[0]

opt_bounds = [
    bounds['F121 CLO circulation flow'],
    bounds['F121outlet temperature'],
    bounds['F121 Oxygen content %']
]
initial_guess = [(opt_bounds[i][0] + opt_bounds[i][1])/2 for i in range(3)]

res = minimize(objective_func, initial_guess, bounds=opt_bounds, method='SLSQP')
best_flow, best_temp, best_oxy = res.x[0], res.x[1], res.x[2]

# --- 5. 預測最佳操作下的 C122 溫度 ---
best_features = np.array([[input_dt, input_c141, best_flow, best_temp, best_oxy]])
predicted_c122_temp = model_c122.predict(best_features)[0]

# --- 6. 主要內容區：顯示最佳化與預測結果 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 系統推薦最佳操作參數")
    st.metric(label="🔹 F121 CLO circulation flow (最佳流量)", value=f"{best_flow:.3f}")
    st.metric(label="🔹 F121outlet temperature (最佳出口溫度)", value=f"{best_temp:.2f} °C")
    st.metric(label="🔹 F121 Oxygen content % (最佳含氧量)", value=f"{best_oxy:.2f} %")

with col2:
    st.subheader("📊 預估效益與製程監控")
    st.info(f"✨ 在目前的排程下，預估最低天然氣消耗量 Y 為： **{res.fun:.2f}**")
    st.success(f"🌡️ 此最佳操作狀態下，預估的 **C122 bottom temperature** 為： **{predicted_c122_temp:.2f} °C**")

# --- 7. 互動式測試：手動微調 ---
st.markdown("---")
st.subheader("🎮 手動操作與即時溫度/能耗連動模擬器")

c_flow = st.slider("手動調整 CLO flow", bounds['F121 CLO circulation flow'][0], bounds['F121 CLO circulation flow'][1], float(best_flow))
c_temp = st.slider("手動調整 出口溫度", bounds['F121outlet temperature'][0], bounds['F121outlet temperature'][1], float(best_temp))
c_oxy = st.slider("手動調整 含氧量 %", bounds['F121 Oxygen content %'][0], bounds['F121 Oxygen content %'][1], float(best_oxy))

manual_features = np.array([[input_dt, input_c141, c_flow, c_temp, c_oxy]])
manual_y = model_ng.predict(manual_features)[0]
manual_c122 = model_c122.predict(manual_features)[0]

res_col1, res_col2 = st.columns(2)
res_col1.metric(label="🏃 手動設定下的預估天然氣消耗 (Y)", value=f"{manual_y:.2f}")
res_col2.metric(label="🌡️ 手動設定下的預估 C122 塔底溫度", value=f"{manual_c122:.2f} °C")import streamlit as st
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
    st.header("💡 隨機森林迴歸（Random Forest Regressor）,30 棵決策樹組成")
    st.info(f"📌 **當前給定條件基準**：DT={input_dt:.4f} | C141={input_c141:.2f} | 出口溫度={input_outlet:.2f} °C")
    
    col1, col2 = st.columns(2)
    with col1: st.metric(label="👉 建議 F121 CLO flow (預測最佳值)", value=f"{opt_flow:.2f}")
    with col2: st.metric(label="👉 建議 F121 含氧量 % (預測最佳值)", value=f"{opt_oxy:.2f} %")
        
    st.markdown("### 📈 F121 NG用量與C122 底部溫度預測")
    st.success(f"🔥 預估最低 **F121 NG consumption (能耗)**: **{min_ng:.2f}**")
    st.warning(f"🌡️ 預估此時 **C122 bottom temperature**: **{predicted_c122_temp:.2f} °C**")
else:
    st.info("👈 請在左側輸入當前的 `DT`、`C141` 與 `F121 outlet temperature`，然後點擊「開始計算最優操作參數」。")
