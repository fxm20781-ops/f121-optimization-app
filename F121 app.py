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
    df = pd.read_excel("data.csv", skiprows=[1], engine="openpyxl")
    
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

    # 【重要變更】特徵（X）的順序保持一致，但模型訓練時 outlet_temp 現在將由使用者在前端給定
    features = [col_dt, col_c141, col_clo, col_outlet, col_oxy]
    X = df[features]
    y_ng = df[col_ng]
    y_c122 = df[col_c122]
    
    # 使用隨機森林模型訓練
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
    return model_ng, model_c122, ranges, features

# 執行載入
try:
    with st.spinner('📊 AI 正在解析數據並建立製程預測模型...'):
        model_ng, model_c122, r, features = load_data_and_train_models()
    st.success('✅ 數據加載成功！智慧優化系統已就緒。')
except Exception as e:
    st.error(f"❌ 數據初始化失敗，錯誤原因: {e}")
    st.stop()

# 2. 側邊欄輸入（已將 F121 outlet temperature 移動到這裡）
st.sidebar.header("📥 輸入當前給定條件")
input_dt = st.sidebar.number_input("1. DT operation", min_value=r['dt_min'], max_value=r['dt_max'], value=0.8000, format="%.4f")
input_c141 = st.sidebar.number_input("2. C141 operation", min_value=r['c141_min'], max_value=r['c141_max'], value=1.48, format="%.2f")
input_outlet = st.sidebar.number_input("3. F121 outlet temperature (°C)", min_value=r['temp_min'], max_value=r['temp_max'], value=(r['temp_min'] + r['temp_max'])/2, format="%.2f")

# 3. 核心最佳化運算
if st.sidebar.button("🚀 開始計算最優操作參數", type="primary"):
    with st.spinner('🔄 正在固定出口溫度，精準尋找最低能耗操作點...'):
        # 調整為對 CLO 流量與含氧量進行網格搜尋（切分 8 點，共 64 種組合）
        flows = np.linspace(r['flow_min'], r['flow_max'], 8)
        oxys = np.linspace(r['oxy_min'], r['oxy_max'], 8)
        
        grid = []
        for f in flows:
            for o in oxys:
                # 這裡要嚴格對照 features 欄位的順序：[DT, C141, CLO, outlet, oxy]
                grid.append([input_dt, input_c141, f, input_outlet, o])
                    
        sim_df = pd.DataFrame(grid, columns=features)
        sim_df['pred_NG'] = model_ng.predict(sim_df)
        
        best_index = sim_df['pred_NG'].idxmin()
        best_run = sim_df.loc[[best_index]].copy()
        predicted_c122_temp = model_c122.predict(best_run[features])[0]
        
        opt_flow = best_run[features[2]].values[0]
        opt_oxy = best_run[features[4]].values[0]
        min_ng = best_run['pred_NG'].values[0]

    # 4. 顯示結果
    st.markdown("---")
    st.header("💡 系統優化引導報告")
    st.info(f"📌 **在此給定條件下**：DT={input_dt:.4f} | C141={input_c141:.2f} | 出口溫度={input_outlet:.2f} °C")
    
    col1, col2 = st.columns(2)
    with col1: st.metric(label="👉 建議 F121 CLO flow", value=f"{opt_flow:.2f}")
    with col2: st.metric(label="👉 建議 F121 含氧量 %", value=f"{opt_oxy:.2f} %")
        
    st.markdown("### 📈 預期效益與副反應預測")
    st.success(f"🔥 預估最低 **F121 NG consumption (能耗)**: **{min_ng:.2f}**")
    st.warning(f"🌡️ 預估此時 **C122 bottom temperature**: **{predicted_c122_temp:.2f} °C**")
else:
    st.info("👈 請在左側輸入當前的 `DT`、`C141` 與 `F121 outlet temperature`，然後點擊「開始計算最優操作參數」。")
