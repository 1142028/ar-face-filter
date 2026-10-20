# AR 臉部濾鏡（OpenCV + MediaPipe）

2025 年 12 月的作品。這個程式會開啟網路攝影機，用 MediaPipe Face Landmarker 抓出臉部特徵點，再把貼圖即時疊到臉上。

## 執行方式

```bash
pip install -r requirements.txt
python 1142028.py
```

按 `q` 結束。

> 第二版的「頭後方貼圖」需要一張去背的 PNG，請命名為 `JOJO.png` 放在同一個資料夾。原本用的是動畫角色圖，有版權問題，所以沒有放進 repo。
> `face_landmarker.task` 是 Google MediaPipe 官方提供的模型（Apache 2.0）。

## 兩個版本

1. **墨鏡濾鏡**：取兩眼外側特徵點（226、350），用兩點的距離決定墨鏡寬度，用 `atan2` 算出頭部傾斜角，把貼圖旋轉後依 alpha 通道疊上。另外也試了用 `solvePnP` 估計頭部姿態。
2. **頭後方貼圖**：想讓圖片看起來在「頭的後面」。做法是先把貼圖畫上去，再用臉部特徵點的凸包（`convexHull`）做遮罩，把原本的人臉像素蓋回來，人臉就會在貼圖前面。

## 現在回頭看

- 遮罩只用臉部特徵點的凸包，所以頭髮和脖子還是會被貼圖蓋住。要做到更自然，需要人像分割（例如 MediaPipe Selfie Segmentation）。
- 檔名是我的學號，保留當時的命名。
