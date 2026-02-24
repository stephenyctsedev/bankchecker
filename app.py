import streamlit as st
import os
import requests
import json
import pandas as pd
import re
from glob import glob
from docling.document_converter import DocumentConverter
import tkinter as tk
from tkinter import filedialog

# --- 初始化頁面設定 ---
st.set_page_config(page_title="銀行利息自動提取器", page_icon="💰", layout="wide")

# --- Helper Functions ---
def get_ollama_models(base_url):
    """從 Ollama API 攞現有嘅模型清單"""
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = [m['name'] for m in response.json()['models']]
            return models
        return ["llama3:8b", "llama3.2"]
    except:
        return ["連線失敗，請檢查 URL"]

def select_folder():
    """彈出視窗俾用家揀 Folder (適用於 Local 執行)"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_selected = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_selected

# --- UI 介面 ---
st.title("💰 銀行月結單利息自動提取器")
st.markdown("透過 **Docling** 解析 PDF 並使用 **Local LLM** 進行數據匯總。")

with st.sidebar:
    st.header("⚙️ 設定")
    ollama_ip = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")
    
    # 動態獲取模型清單
    model_list = get_ollama_models(ollama_ip)
    selected_model = st.selectbox("選擇 LLM 模型", model_list)
    
    st.divider()
    if st.button("📁 選擇月結單資料夾"):
        folder_path = select_folder()
        st.session_state['folder_path'] = folder_path

# 顯示已選路徑
current_folder = st.session_state.get('folder_path', "未選擇資料夾")
st.info(f"📍 當前處理路徑: `{current_folder}`")

# --- 核心邏輯 ---
if st.button("🚀 開始掃描並轉換", type="primary"):
    if not os.path.exists(current_folder) or current_folder == "未選擇資料夾":
        st.error("請先選擇一個有效的資料夾！")
    else:
        pdf_files = glob(os.path.join(current_folder, "*.pdf"))
        if not pdf_files:
            st.warning("資料夾內冇 PDF 檔案。")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_results = []
            
            converter = DocumentConverter()
            
            for idx, pdf in enumerate(pdf_files):
                filename = os.path.basename(pdf)
                status_text.text(f"正在處理 ({idx+1}/{len(pdf_files)}): {filename}")
                
                # 1. Docling 轉換
                result = converter.convert(pdf)
                md_text = result.document.export_to_markdown()
                
                # 2. Call Ollama
                payload = {
                    "model": selected_model,
                    "prompt": f"請從以下文本中提取所有利息收入項目，以 JSON 格式輸出：[{{'date': 'YYYY/MM/DD', 'description': '名稱', 'amount': 0.00}}]。文本：\n{md_text}",
                    "stream": False,
                    "format": "json"
                }
                
                try:
                    res = requests.post(f"{ollama_ip}/api/generate", json=payload)
                    response_data = res.json().get('response', '[]').strip()
                    
                    # --- 強效清理步驟 ---
                    # 1. 移除 JSON 以外嘅文字 (有時 LLM 會加 "Here is your JSON:")
                    json_match = re.search(r'\[.*\]', response_data, re.DOTALL)
                    if json_match:
                        clean_json = json_match.group(0)
                    else:
                        clean_json = response_data

                    # 2. 處理常見語法錯誤：將單引號轉雙引號，移除尾隨逗號
                    clean_json = clean_json.replace("'", '"')
                    clean_json = re.sub(r',\s*\]', ']', clean_json) # 移除 [...,] 嘅逗號
                    
                    # 嘗試解析
                    items = json.loads(clean_json)
                    
                    # --- 防錯檢查：確保 items 係一個 list ---
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict): # 確保入面係字典
                                item['source'] = filename  # 呢度就係原本出錯嘅地方
                                all_results.append(item)
                    elif isinstance(items, dict): # 有時模型只會回傳單一物件
                        items['source'] = filename
                        all_results.append(items)
                        
                except Exception as e:
                    st.error(f"分析 {filename} 時出錯: {e}")
                    # 打印出嚟睇吓 LLM 到底俾咗咩你，方便除錯
                    st.code(response_data, language="json")            
                progress_bar.progress((idx + 1) / len(pdf_files))

            # 3. 顯示結果
            if all_results:
                df = pd.DataFrame(all_results)
                
                # --- 新增：欄位清洗機制 ---
                # 預防模型俾錯名 (例如 '金額' -> 'amount')
                rename_map = {
                    '金額': 'amount', 
                    'Value': 'amount', 
                    'price': 'amount',
                    '日期': 'date',
                    'description': 'description',
                    '項目': 'description'
                }
                df.rename(columns=rename_map, inplace=True)

                # 檢查 'amount' 欄位是否存在
                if 'amount' in df.columns:
                    # 去除數字入面的千分位逗號 (例如 1,234.50 -> 1234.50)
                    df['amount'] = df['amount'].astype(str).str.replace(',', '').str.replace('$', '')
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
                else:
                    # 如果真係冇 amount 欄位，就補一個全 0 嘅俾佢，防止報錯
                    df['amount'] = 0.0

                st.success("✅ 處理完成！")
                st.subheader("📊 利息收入匯總表")
                st.dataframe(df, use_container_width=True)
                
                total = df['amount'].sum()
                st.metric("全年總利息收入", f"${total:,.2f}")