import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="F121 製程操作最佳化系統", layout="wide")
st.title("🏭 F121 加熱爐操作最佳化與預測系統")

# 1. 讀取數據與模型訓練（強效快取）
@st.cache_resource
def load_data_and_train_models():
    # 【最暴力的解法】直接丟棄 CSV 的前兩行表頭 (skiprows=2)，重新賦予標準欄位名稱
    # 這樣管它 CSV 裡面有幾個空格、換行符號還是代號，進到 Python 通通變標準名稱！
    df = pd.read_csv("data.csv", skiprows=2, header=None).dropna()
    
    # 依據您原始 CSV 的 9 個欄位順序，由左至右精準命名：
    # 0: 時間, 1: DT, 2: C141, 3: C122溫度, 4: C121溫度, 5: CLO流量, 6: 出口溫度, 7: NG能耗, 8: 含氧量
    df.columns = [
        'Time', 'DT', 'C141', 'C122_temp', 'C121_temp', 
        'CLO_flow', 'outlet_temp', 'NG_consumption', 'oxygen_content'
    ]
    
    # 標準特徵與目標
    features = ['DT', 'C141', 'CLO_flow', 'outlet_temp', 'oxygen_content']
    X = df[features]
    y_ng = df['NG_consumption']
    y_c122 = df['C122_temp']
    
    # 使用極速隨機森林（30棵樹，確保雲端不超時）
    model_ng = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
    model_ng.fit(X, y_ng)
    
    model_c122 = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
    model_c122.fit(X, y_c122)
    
    ranges = {
        'dt_min': float(df['DT'].min()), 'dt_max': float(df['DT'].max()),
        'c141_min': float(df['C141'].min()), 'c141_max': float(df['C141'].max()),
        'flow_min': float(df['CLO_flow'].min()), 'flow_max': float(df['CLO_flow'].max()),
        'temp_min': float(df['outlet_temp'].min()), 'temp_max': float(df['outlet_temp'].max()),
        'oxy_min': float(df['oxygen_content'].min()), 'oxy_max': float(df['oxygen_content'].max()),
    }
    return model_ng, model_c122, ranges, features

# 執行載入
try:
    with st.spinner('📊 正在載入數據並建立 AI 預測模型...'):
        model_ng, model_c122, r, features = load_data_and_train_models()
    st.success('✅ 模型載入完成，系統已準備就緒！')
except Exception as e:
    st.error(f"❌ 數據初始化失敗，錯誤原因: {e}")
    st.stop()

# 2. 側邊欄輸入
st.sidebar.header("📥 輸入當前製程條件")
input_dt = st.sidebar.number_input("1. DT operation", min_value=r['dt_min'], max_value=r['dt_max'], value=0.8000, format="%.4f")
input_c141 = st.sidebar.number_input("2. C141 operation", min_value=r['c141_min'], max_value=r['c141_max'], value=1.48, format="%.2f")

# 3. 核心最佳化運算
if st.sidebar.button("🚀 開始計算最優操作參數", type="primary"):
    with st.spinner('🔄 正在精準計算最低能耗解...'):
        # 8點切分（共512種組合）
        flows = np.linspace(r['flow_min'], r['flow_max'], 8)
        temps = np.linspace(r['temp_min'], r['temp_max'], 8)
        oxys = np.linspace(r['oxy_min'], r['oxy_max'], 8)
        
        grid = []
        for f in flows:
            for t in temps:
                for o in oxys:
                    grid.append([input_dt, input_c141, f, t, o])
                    
        sim_df = pd.DataFrame(grid, columns=features)
        sim_df['pred_NG'] = model_ng.predict(sim_df)
        
        best_index = sim_df['pred_NG'].idxmin()
        best_run = sim_df.loc[[best_index]].copy()
        predicted_c122_temp = model_c122.predict(best_run[features])[0]
        
        opt_flow = best_run['CLO_flow'].values[0]
        opt_temp = best_run['outlet_temp'].values[0]
        opt_oxy = best_run['oxygen_content'].values[0]
        min_ng = best_run['pred_NG'].values[0]

    # 4. 顯示結果
    st.markdown("---")
    st.header("💡 系統優化引導報告")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="👉 建議 F121 CLO flow", value=f"{opt_flow:.2f}")
    with col2: st.metric(label="👉 建議 F121 出口溫度", value=f"{opt_temp:.2f} °C")
    with col3: st.metric(label="👉 建議 F121 含氧量 %", value=f"{opt_oxy:.2f} %")
        
    st.markdown("### 📈 預期效益與副反應預測")
    st.info(f"🔥 預估最低 **F121 NG consumption (能耗)**: **{min_ng:.2f}**")
    st.warning(f"🌡️ 預估此時 **C122 bottom temperature**: **{predicted_c122_temp:.2f} °C**")
else:
    st.info("👈 請在左側輸入當前的 `DT operation` 與 `C141 operation`，然後點擊「開始計算最優操作參數」。")
