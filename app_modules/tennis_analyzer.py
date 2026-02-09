import cv2
import numpy as np
import math
import time
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------- CONFIG ----------------------
VIDEO_PATH = "tennis.mp4"
OUTPUT_PATH = "tennis_analytics.mp4"
YOLO_MODEL = "yolov8n.pt"
POSE_MODEL = "models/pose_landmarker_heavy.task"
PIXELS_PER_METER = 50
HISTORY_LENGTH = 3
SPEED_HISTORY_LENGTH = 50
# ----------------------------------------------------

# Load models
yolo = YOLO(YOLO_MODEL)
base = python.BaseOptions(model_asset_path=POSE_MODEL)
pose_opts = vision.PoseLandmarkerOptions(base_options=base, output_segmentation_masks=False, num_poses=1)
pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)

# Data trackers
ball_hist = deque(maxlen=HISTORY_LENGTH)
racket_hist = deque(maxlen=HISTORY_LENGTH)
speed_hist = deque(maxlen=SPEED_HISTORY_LENGTH)
last_pos = {"sports ball": None, "tennis racket": None}
last_t = None
elbow_hist = deque(maxlen=30)
curr_speed = None

# Helpers
def dist(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])
def calc_angle(a,b,c):
    a, b, c = map(np.array,[a,b,c])
    ab, cb = a-b, c-b
    dot = ab.dot(cb)
    mag = np.linalg.norm(ab)*np.linalg.norm(cb)
    if not mag: return 0
    return int(math.degrees(math.acos(np.clip(dot/mag, -1, 1))))
def swing_phase(hist):
    if len(hist)<5: return "Idle"
    r = list(hist)[-5:]
    return "Forward Swing" if max(r)-min(r)>40 else "Follow-Through" if np.mean(r)<70 else "Backswing"
def draw_arc(img, A,B,C,angle,color=(0,255,0),r=40):
    v1,v2 = np.array(A)-np.array(B), np.array(C)-np.array(B)
    sa, ea = math.degrees(math.atan2(v1[1],v1[0])), math.degrees(math.atan2(v2[1],v2[0]))
    cv2.ellipse(img, B, (r,r), 0, sa, ea, color, 2)
    cv2.putText(img, f"{angle}°", (B[0]+5, B[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,2)
def draw_speed_chart(img, hist):
    h,w = 50,200
    y0,x0 = 10, img.shape[1] - w - 10
    cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (50,50,50), -1)
    if not hist: return
    maxv = max(max(hist),1)
    pts = [(x0 + int(i*(w/len(hist))), y0 + h - int((v/maxv)*h)) for i,v in enumerate(hist)]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], (0,255,0), 2)

# Main
def start_tennis_analysis():
    global last_t, curr_speed
    cap = cv2.VideoCapture(VIDEO_PATH)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    while True:
        ret, frame = cap.read()
        if not ret: break
        t = time.time()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prs = pose_detector.detect(mp_img)

        # Pose & Angle Visualization
        if prs.pose_landmarks:
            lm = prs.pose_landmarks[0]
            def gp(i): return (int(lm[i].x*w), int(lm[i].y*h))
            S, E, W = gp(12), gp(14), gp(16)
            H, K, A = gp(24), gp(26), gp(28)
            for p,q in [(S,E),(E,W),(H,K),(K,A)]: cv2.line(frame, p,q, (0,255,0),2)
            ea = calc_angle(S,E,W); elbow_hist.append(ea); draw_arc(frame,S,E,W,ea)
            ka = calc_angle(H,K,A); draw_arc(frame,H,K,A,ka,(255,0,0))
            cv2.putText(frame, swing_phase(elbow_hist), (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        # Object Detection & Trail, Speed
        dets = yolo(frame, verbose=False)
        for res in dets:
            for b in res.boxes:
                label = yolo.names[int(b.cls[0])].lower()
                if label in ["sports ball","tennis racket"]:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    col = (0,255,255) if "ball" in label else (255,0,0)
                    cv2.circle(frame, (cx,cy), 8, col, -1); cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
                    if last_pos.get(label) and last_t:
                        d, dt = dist((cx,cy), last_pos[label]), t-last_t
                        sp = (d/PIXELS_PER_METER)/dt if dt>1e-3 else 0
                        curr_speed = sp*3.6
                        speed_hist.append(curr_speed)
                        # speed arrow for ball
                        if "sports ball" in label:
                            lp = last_pos[label]
                            cv2.arrowedLine(frame, lp, (cx,cy), (0,255,255),2, tipLength=0.2)
                    last_pos[label] = (cx,cy)
                    if "ball" in label: ball_hist.append((cx,cy))
                    else: racket_hist.append((cx,cy))
        last_t = t

        # Draw trails with acceleration color for racket
        for i in range(1,len(racket_hist)):
            p1, p2 = racket_hist[i-1], racket_hist[i]
            if i<2 or not speed_hist: col = (255,0,0)
            else:
                a = abs(speed_hist[-1] - speed_hist[-2])
                cv2.line(frame, p1, p2, (0, int(min(a*50,255)),255-int(min(a*50,255))), 3)
        for i in range(1,len(ball_hist)): cv2.line(frame, ball_hist[i-1], ball_hist[i], (0,255,255),2)

        # Overlay speed & chart
        if curr_speed: cv2.putText(frame,f"Speed: {curr_speed:.1f} km/h",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        draw_speed_chart(frame, speed_hist)

        # Bounce & Topspin
        ys = [p[1] for p in ball_hist]
        if len(ys)>=3 and ys[-3]>ys[-2]<ys[-1]:
            cv2.putText(frame,"Bounce!",(w-200,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        if len(ball_hist)>=5 and sum([ball_hist[i+1][1]-ball_hist[i][1] for i in range(-5,-1)])<0:
            cv2.putText(frame,"Topspin!",(w-200,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        out.write(frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Video saved:", OUTPUT_PATH)


