# OpenCV Project

本專案是一組以 OpenCV 為核心的工具集合，目前包含：

- face 資料擷取（`add_face.py`）
- 人臉模型訓練（`train.py`）
- 即時辨識展示（`face_cog.py`）
- 人臉偵測/展示（`face.py`，demo 用）
- 文件掃描（`main.py`，四點透視掃描/存圖）

目標：提供一個輕量可改造的臉部採集→訓練→辨識流程，以及一個簡單的文件掃描工具，方便快速實驗與原型開發。

## 專案結構（重點）

- `add_face.py`  - 用 webcam 擷取人臉圖片並存入 `face/faceXX/` 資料夾（預設 30 張）。
- `train.py`     - 讀取 `face/face*` 下各個子資料夾作為不同 ID，使用 Haar cascade 偵測臉並訓練 LBPH 模型，輸出 `trainer.yml`。
- `face_cog.py`  - 載入 `trainer.yml` 與 `face.xml` 做即時辨識與畫框顯示（`name` 映射目前為 code 內硬編碼）。
- `face.py`      - 顯示偵測或模糊臉部的 demo 程式（實驗用途）。
- `main.py`      - 文件掃描流程（四點透視、穩定性偵測、對焦檢查），會將結果存到 `scan/`。
- `face.xml`     - Haar cascade 檔案（臉部偵測）。
- `trainer.yml`  - 訓練完的模型（若存在）。
- `requirements.txt` - 目前依賴清單（建議鎖版本）。

## 先決條件

- Python 3.8+（建議 3.10/3.11）
- webcam（或能接入的攝影機）
- 建議安裝 opencv-contrib-python（需要使用 `cv2.face`）

注意：不同 OpenCV 版本可能導致 `cv2.face` 不可用，請安裝 `opencv-contrib-python`。

## 安裝（建議）

在 macOS 的 zsh 下：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 如果 requirements.txt 沒有 contrib，請額外安裝：
pip install opencv-contrib-python
```

如果你遇到 OpenCV 與 contrib 相容問題，可以改用指定版本，例如：

```bash
pip install opencv-python==4.5.5.62 opencv-contrib-python==4.5.5.62
```

## 快速上手（3 個步驟）

1. 擷取臉部圖片（執行 `add_face.py`）

```bash
python add_face.py
```

流程：程式會要求你輸入 ID（目前預期為兩位數格式，如 `01`），會在 `face/face01/` 下存 30 張灰階人臉圖片。若攝影機不是 0，請改成 `--camera`（見下方參數化建議）。

2. 訓練模型（執行 `train.py`）

```bash
python train.py
```

流程：會掃描 `face/` 目錄下的 `faceXX` 資料夾，對每張圖做 Haar 偵測，收集 faces 與 ids，訓練 LBPH 模型並輸出 `trainer.yml`。

3. 即時辨識（執行 `face_cog.py`）

```bash
python face_cog.py
```

流程：載入 `trainer.yml` 與 `face.xml`，開啟 webcam 進行辨識並在畫面上顯示名稱（目前名稱表寫在程式內）。按 `q` 可以退出視窗。

## 推薦參數化（未實作但建議）

為了更靈活使用，建議把下列支援加到腳本：

- `--camera`：攝影機編號（0, 1, ...）
- `--out-dir`：輸出人臉資料夾（預設 `face/`）
- `--count`：要擷取的影像數量（預設 30）
- `--cascade`、`--model`：cascade 路徑與模型檔案
- 讓 `face_cog.py` 的 id→name 映射從 `face/` 資料夾自動產生或讀 `names.json`

## 故障排除（常見）

- 無法開啟攝影機：確認系統授權、使用 `ls /dev` 檢查裝置 index，或把 `cap = cv2.VideoCapture(1)` 改成其他 index。
- 找不到 `cv2.face`：請安裝 `opencv-contrib-python`。若仍錯誤，使用相容的指定版本安裝。
- 偵測不到臉：確保光線充足，face cascade 參數（scaleFactor/minNeighbors）可調。

## 改善與下一步（TODO 建議）

短期（High priority）
- 補 README（完成）
- 讓 `add_face.py`, `train.py`, `face_cog.py` 支援 argparse 參數化（camera, out-dir, count）
- 自動化 `names.json` 生成功能（從 `face/` 資料夾讀取）

中期（Medium）
- 建 `retrain.py` 或 `scripts/retrain.sh` 自動化訓練流程
- 加入資料擴增提升模型健壯性
- 改用更先進的偵測/辨識方法（DNN 人臉偵測、embedding + classifier）

長期（Low）
- 單元測試（pytest）與 CI pipeline
- 整合 `scan` 與 `face` 功能到一個 CLI（`cli.py`）

## 小技巧

- 若想快速看人臉檔名對應，可以執行：

```bash
ls face | sort
# 或產生 names.json
python - <<'PY'
import json,os
names = {i:os.listdir('face')[i] for i in range(len(os.listdir('face')))}
print(json.dumps(names, ensure_ascii=False, indent=2))
PY
```

## 聯絡與協作

如果你要我接著把 README 裡建議的「參數化 `add_face.py`」或「自動生成 `names.json`」實作，我可以立刻修改並執行 smoke test。告訴我你要我下一步做哪一項（例如：參數化 `add_face.py`）。

---

最後更新：2025-10-04
