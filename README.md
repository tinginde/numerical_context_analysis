# 數值語境表示分析 / Contextual Numerical Representation Analysis

研究同一個數字在不同語境下，LLM 的內部表示（hidden states）是否有差異。

---

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `numerical_context_analysis_v1.py` | 基礎版：研究 "24" 在不同語境（年齡、溫度、時間等）下的 hidden state 差異 |
| `numerical_context_analysis_v1.png` | v1 輸出圖 |
| `numerical_context_analysis_v2.py` | 進階版：新增實驗 C，研究相同語境但不同危險程度的醫療數值（血壓、體溫、心跳等） |
| `numerical_context_analysis_v2.png` | v2 輸出圖 |
| `requirements.txt` | 環境依賴清單 |

---

## 環境安裝

```bash
pip install -r requirements.txt
```

---

## HuggingFace 登入

程式使用 Meta LLaMA 模型，需要先登入 HuggingFace（**金鑰不包含在此資料夾中**，請自行設定）：

```bash
huggingface-cli login
```

登入後憑證會存在本機 `~/.cache/huggingface/`，程式執行時自動讀取。

---

## 執行方式

```bash
# v1：基礎語境實驗
python numerical_context_analysis_v1.py

# v2：醫療數值語境實驗
python numerical_context_analysis_v2.py
```

---

## 實驗設計

### v1 — 跨語境數字表示
- **實驗 A**：同一個數字 "24" 放在不同語境句子（年齡、溫度、時間、距離、金錢）
- **實驗 B**：不同數量級的數字（2、24、240）放在相同語境

### v2 — 醫療語境危險程度辨識（新增）
- **實驗 C**：相同語境，數字代表截然不同的臨床意義
  - 血壓 (mmHg)：40 / 79 / 120 / 180
  - 體溫 (°C)：25 / 35 / 37 / 41
  - 心跳 (bpm)：30 / 55 / 75 / 180
  - 血糖 (mg/dL)：40 / 90 / 180 / 400
  - 呼吸頻率 (次/分)：4 / 16 / 25 / 40
