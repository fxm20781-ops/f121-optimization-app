import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def run_optimization_program(dt_operation_input, c141_operation_input, given_f121_outlet_temp):
    # 1. 讀取數據（排除第二行單位標籤並清除欄位名稱前後空格與換行）
    file_path = 'data.csv' # Assumes file is in the same directory as the script
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 '{file_path}'。請確保檔案位於程式碼相同的目錄下。")
        return

    df = pd.read_excel(file_path, skiprows=[1])
    df.columns = df.columns.str.replace('\n', '').str.strip()

    # Ensure 'DT operation' and 'C141 operation' are numeric before using them in features
    df['DT operation'] = pd.to_numeric(df['DT operation'], errors='coerce')
    df['C141 operation'] = pd.to_numeric(df['C141 operation'], errors='coerce')

    # 2. 定義模型特徵與目標
    # F121outlet temperature 現在是給定條件，但模型仍需使用它進行預測
    features = [
        'DT operation', 'C141 operation',
        'F121 CLO circulation flow', 'F121outlet temperature', 'F121  Oxygen content %'
    ]

    target_ng = 'F121 NG consumption'
    target_c122 = 'C122 bottom temperature'

    # Convert target columns to numeric, coercing errors to NaN
    df[target_ng] = pd.to_numeric(df[target_ng], errors='coerce')
    df[target_c122] = pd.to_numeric(df[target_c122], errors='coerce')

    # Ensure all features and targets are numeric and drop rows with NaNs in these critical columns
    df_model = df.dropna(subset=features + [target_ng, target_c122]).copy()

    X = df_model[features]
    y_ng = df_model[target_ng]
    y_c122 = df_model[target_c122]

    # 3. 訓練兩個隨機森林模型
    st.write("正在訓練製程最佳化模型與溫度預測模型...")
    model_ng = RandomForestRegressor(n_estimators=100, random_state=42)
    model_ng.fit(X, y_ng)

    model_c122 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_c122.fit(X, y_c122)
    st.write("模型訓練完成！")

    # 4. 定義核心尋優與預測函數
    # 使用使用者指定的操作範圍
    flow_min, flow_max = 50, 55 # F121 CLO circulation flow 範圍 50~55
    # F121outlet temperature 範圍 330~340 現在是給定值，不再優化
    oxy_min, oxy_max = 4, 7 # F121 Oxygen content % 範圍 4~7%

    # 建立網格搜索空間（各切割 25 個點，共 25 x 25 = 625 種可能的操作組合）
    flows = np.linspace(flow_min, flow_max, 25)
    oxys = np.linspace(oxy_min, oxy_max, 25)

    grid = []
    for f in flows:
        for o in oxys:
            # F121outlet temperature 現在是固定輸入值
            grid.append([dt_operation_input, c141_operation_input, f, given_f121_outlet_temp, o])

    sim_df = pd.DataFrame(grid, columns=features)

    # 使用模型 A 預測這萬種操作組合在當前 DT/C141 下的天然氣消耗量
    sim_df['pred_NG'] = model_ng.predict(sim_df)

    # 找出預測能耗最小（MIN）的那一組最優操作參數
    best_index = sim_df['pred_NG'].idxmin()
    best_run = sim_df.loc[[best_index]].copy()

    # 使用模型 B，針對這組「最優操作參數」預測此時的 C122 bottom temperature
    predicted_c122_temp = model_c122.predict(best_run[features])[0]

    # 整理結果輸出
    opt_flow = best_run['F121 CLO circulation flow'].values[0]
    # F121outlet temperature 現在是給定值，不再是優化結果
    opt_oxy = best_run['F121  Oxygen content %'].values[0]
    min_ng = best_run['pred_NG'].values[0]
    st.write("\n================== 系統最佳化引導報告 ==================")
    st.write(f"【目前給定不可控條件】:")
    st.write(f"  - DT operation           : {dt_operation_input:.4f}")
    st.write(f"  - C141 operation         : {c141_operation_input:.4f}")
    st.write(f"  - F121outlet temperature : {given_f121_outlet_temp:.2f}") # 將其列為給定條件
    st.write(f"\n【建議之最優操作控制參數（目標：最低 NG 能耗）】:")
    st.write(f"  👉 F121 CLO circulation flow : {opt_flow:.2f}")
    st.write(f"  👉 F121  Oxygen content %    : {opt_oxy:.2f}")
    st.write(f"\n【預期操作結果與影響預測】:")
    st.write(f"  - 預估最低 F121 NG consumption   : {min_ng:.2f}")
    st.write(f"  - 預估此時 C122 bottom temperature: {predicted_c122_temp:.2f} (⚠️請確認此溫度是否在安全操作區間)")
    st.write("=====================================================")


# --- 5. 現場操作模擬測試 ---
# 您可以將以下數值修改為目前製程當下的實際 DT 與 C141 數值
if __name__ == "__main__":
    st.write("歡迎使用製程最佳化程式！")
    st.write("請輸入當前的不可控條件：")

    try:
        current_dt_input = st.sidebar.number_input("  - 請輸入 DT operation: "))
        current_c141_input = st.sidebar.number_input("  - 請輸入 C141 operation: "))
        current_f121_outlet_temp_input =st.sidebar.number_input("  - 請輸入 F121outlet temperature: ")) # 新增此輸入
    except ValueError:
        st.write("輸入無效。請輸入數字。")
    else:
        run_optimization_program(current_dt_input, current_c141_input, current_f121_outlet_temp_input)
