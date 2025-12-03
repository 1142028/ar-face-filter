import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math # 用於數學計算，如弧度轉換

# --- 設定路徑 ---
MODEL_PATH = 'face_landmarker.task'
SUNGLASSES_PATH = 'sunglasses.png'

# --- 1. 影像處理工具函式 ---

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """
    將帶有透明度的圖片 (img_overlay) 疊加到主圖片 (img) 上。
    :param img: 主圖片 (BGR)
    :param img_overlay: 要疊加的圖片 (BGR)
    :param pos: 疊加圖片的左上角位置 (x, y)
    :param alpha_mask: 透明度遮罩 (0.0 到 1.0)
    """
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1_overlay, y2_overlay = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1_overlay, x2_overlay = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if x1 >= x2 or y1 >= y2:
        return img

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    alpha = alpha_mask[y1_overlay:y2_overlay, x1_overlay:x2_overlay, np.newaxis]
    
    img_crop[:] = alpha * img_overlay_crop + (1.0 - alpha) * img_crop
    
    return img

def rotate_image(image, angle, center=None, scale=1.0):
    """
    旋轉圖片，同時調整大小以避免裁切。
    """
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # 取得旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 計算旋轉後的新圖片邊界
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 調整旋轉矩陣的平移部分，使其能包含整個圖片
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    return cv2.warpAffine(image, M, (nW, nH))


def apply_sunglasses_filter(frame, landmarks, sunglasses_img, W, H):
    """
    根據臉部特徵點，計算並繪製帶有旋轉功能的墨鏡貼圖。
    """
    # MediaPipe Face Landmarker 關鍵特徵點索引 (用於墨鏡定位):
    # 左眼外側: 226
    # 右眼外側: 350
    # 左眼內側: 113
    # 右眼內側: 340
    # 鼻尖: 4
    # 嘴巴中心: 13
    # 左嘴角: 61
    # 右嘴角: 291

    # 1. 取得臉部關鍵點的 2D 像素座標
    # 我們需要多一些點來估計頭部姿態
    points_2d = []
    
    # 左右眼外側
    x_left = int(landmarks[226].x * W)
    y_left = int(landmarks[226].y * H)
    x_right = int(landmarks[350].x * W)
    y_right = int(landmarks[350].y * H)
    
    points_2d.append([x_left, y_left]) # 左眼外側
    points_2d.append([x_right, y_right]) # 右眼外側
    
    # 鼻尖
    nose_tip_x = int(landmarks[4].x * W)
    nose_tip_y = int(landmarks[4].y * H)
    points_2d.append([nose_tip_x, nose_tip_y]) # 鼻尖

    # 嘴巴中心上方
    mouth_center_x = int(landmarks[13].x * W)
    mouth_center_y = int(landmarks[13].y * H)
    points_2d.append([mouth_center_x, mouth_center_y]) # 嘴巴中心

    # 左右嘴角
    left_mouth_x = int(landmarks[61].x * W)
    left_mouth_y = int(landmarks[61].y * H)
    right_mouth_x = int(landmarks[291].x * W)
    right_mouth_y = int(landmarks[291].y * H)
    points_2d.append([left_mouth_x, left_mouth_y]) # 左嘴角
    points_2d.append([right_mouth_x, right_mouth_y]) # 右嘴角
    
    points_2d = np.array(points_2d, dtype=np.float64)

    # 2. 定義對應的 3D 臉部模型點 (這些是估計值，代表臉部在 3D 空間中的標準位置)
    # 單位可以是任意的，只要保持相對比例正確即可
    # 我們可以假設：
    #      (0, 0, 0) 是鼻尖
    #      眼睛在鼻尖的 Y 軸上方，X 軸左右
    #      嘴巴在鼻尖的 Y 軸下方
    
    model_points_3d = np.array([
        (-70.0, 30.0, -30.0),   # 左眼外側 (估計值)
        (70.0, 30.0, -30.0),    # 右眼外側 (估計值)
        (0.0, 0.0, 0.0),        # 鼻尖 (作為原點)
        (0.0, -30.0, 10.0),     # 嘴巴中心 (估計值)
        (-40.0, -30.0, 0.0),    # 左嘴角 (估計值)
        (40.0, -30.0, 0.0)      # 右嘴角 (估計值)
    ], dtype=np.float64)

    # 3. 相機內參矩陣 (Camera Matrix) 和畸變係數 (Distortion Coefficients)
    # 我們使用一個簡化的相機矩陣。fx, fy 設為畫面寬度，cx, cy 設為畫面中心。
    focal_length = W  # 近似值
    center = (W / 2, H / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # 假設沒有鏡頭畸變
    dist_coeffs = np.zeros((4, 1)) 

    # 4. 使用 solvePnP 估計頭部姿態
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        # 將旋轉向量轉換為旋轉矩陣 (用於更精確的旋轉角度計算)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 將旋轉矩陣轉換為歐拉角 (Pitch, Yaw, Roll)
        # 這是一個簡化但常用的方法，對於小角度效果良好
        pitch = rotation_matrix[0, 1] # 左右傾斜 (roll)
        yaw = rotation_matrix[0, 2]   # 點頭 (pitch)
        roll = rotation_matrix[1, 2]  # 甩頭 (yaw)

        # 這裡我們主要關心 Z 軸的旋轉，也就是頭部左右轉的 Yaw 角度
        # 或者說，墨鏡在圖片平面上的旋轉角度
        # 最直接影響圖片平面旋轉的是 Roll (左右傾斜)
        
        # 為了簡化，我們直接使用兩個眼睛點的連線角度作為墨鏡的平面旋轉
        # 計算兩眼連線的弧度
        angle_rad = math.atan2(y_right - y_left, x_right - x_left)
        # 轉換為角度，並校正為水平角度 (墨鏡圖片通常是水平的)
        angle_deg = np.degrees(angle_rad) 
        
        angle_deg = -angle_deg

        # 5. 調整墨鏡圖片的旋轉和縮放
        
        # 計算所需的墨鏡寬度 (使用左右眼角距離)
        eye_distance = np.sqrt((x_right - x_left)**2 + (y_right - y_left)**2)
        target_width = int(eye_distance * 2.2) # 墨鏡比眼睛寬一些
        
        # 縮放墨鏡圖片 (根據目標寬度)
        sunglasses_H, sunglasses_W, _ = sunglasses_img.shape
        scale_factor = target_width / sunglasses_W
        
        resized_sunglasses_raw = cv2.resize(sunglasses_img, (target_width, int(sunglasses_H * scale_factor)))
        
        # 旋轉墨鏡圖片 (以墨鏡中心為旋轉點)
        # 這裡的旋轉角度是 `angle_deg`
        rotated_sunglasses = rotate_image(resized_sunglasses_raw, angle_deg) # 使用我們定義的旋轉函式
        
        # 6. 準備 Alpha 遮罩
        sunglasses_bgr = rotated_sunglasses[:, :, :3]
        sunglasses_alpha = rotated_sunglasses[:, :, 3] / 255.0 # 標準化到 0.0-1.0
        
        # 7. 計算疊加位置 (我們將墨鏡中心對齊兩眼中心)
        center_x = (x_left + x_right) // 2
        center_y = (y_left + y_right) // 2
        
        # 新增：水平偏移量 (您可以調整這些值)
        HORIZONTAL_OFFSET = 20  
        VERTICAL_OFFSET = -5   # 向上微調
        
        # 墨鏡的左上角位置
        x_pos = center_x - rotated_sunglasses.shape[1] // 2 + HORIZONTAL_OFFSET
        y_pos = center_y - int(rotated_sunglasses.shape[0] * 0.4) + VERTICAL_OFFSET # 調整垂直位置
        
        # 8. 疊加到主畫面上
        frame = overlay_image_alpha(frame, sunglasses_bgr, (x_pos, y_pos), sunglasses_alpha)
        
    return frame

# --- 2. 初始化 MediaPipe Face Landmarker ---

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

try:
    # 載入墨鏡圖片 (需要處理 PNG 的透明通道)
    sunglasses = cv2.imread(SUNGLASSES_PATH, cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        raise FileNotFoundError(f"無法載入圖片：{SUNGLASSES_PATH}。請檢查路徑和檔案是否存在。")
    if sunglasses.ndim < 3 or sunglasses.shape[2] < 4:
        raise ValueError("載入的墨鏡圖片不是帶有透明度 (4通道) 的 PNG 圖片。請更換檔案。")
    
    # 建立 Face Landmarker 選項
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1  # 追蹤一張臉
    )

    # 建立 Face Landmarker 實例
    landmarker = FaceLandmarker.create_from_options(options)

except FileNotFoundError as e:
    print(f"錯誤：{e}")
    print("請確認您已下載 face_landmarker.task 模型和 sunglasses.png 貼圖，並放在正確的路徑。")
    exit()
except Exception as e:
    print(f"初始化錯誤: {e}")
    exit()


# --- 3. 主循環：影像捕獲與濾鏡應用 ---

cap = cv2.VideoCapture(0) # 0 代表預設的網路攝影機
timestamp = 0

if not cap.isOpened():
    print("無法打開攝影機，請檢查攝影機連線或是否有其他程式佔用。")
    landmarker.close()
    exit()

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape
        
        # 1. 影像預處理：轉換為 RGB 格式供 MediaPipe 使用
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 重要：修正這裡的 ImageFormat
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. 進行臉部特徵點偵測
        timestamp += 1
        face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)

        # 3. 處理偵測結果並應用濾鏡
        if face_landmarker_result.face_landmarks:
            landmarks = face_landmarker_result.face_landmarks[0]
            
            # 呼叫墨鏡繪製函式
            frame = apply_sunglasses_filter(frame, landmarks, sunglasses, W, H) 
        
        # 4. 顯示結果
        cv2.imshow('Python Sunglasses Filter (Press Q to exit)', frame)

        # 按 'q' 鍵退出迴圈
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    # 5. 資源釋放
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("程式已關閉。")