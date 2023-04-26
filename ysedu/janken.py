from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import Image
import math
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
import cv2
from google.colab.patches import cv2_imshow

def 距離(pt1, pt2):
  return math.sqrt(math.pow(pt1.x - pt2.x, 2) + math.pow(pt1.y - pt2.y, 2))

def 距離リスト(pts, ori):
  リスト = []
  for pt in pts:
    リスト.append(距離(pt, ori))
  return リスト

def takePhoto(file, quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();
      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  print(data.split(',')[1])
  # 取り込んだデータをColab上に一時保存
  画像データ = b64decode(data.split(',')[1])
  with open(file, 'wb') as f:
    f.write(画像データ)

def じゃんけん判定(ファイル, 手画像):
  try:
    takePhoto(ファイル)
    # print('Saved to {}'.format(ファイル))
    画像 = cv2.imread(ファイル)
    with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.0) as hands:
        # Convert the BGR image to RGB, flip the image around y-axis for correct 
        # handedness output and process it with MediaPipe Hands.
        検出結果 = hands.process(cv2.flip(cv2.cvtColor(画像, cv2.COLOR_BGR2RGB), 1))
        image_hight, image_width, _ = 画像.shape
        # Print handedness (left v.s. right hand).
        #print(f'Handedness of {name}:')
        # print(results.multi_handedness)
        # Draw hand landmarks of each hand.
        # print(f'Hand landmarks of {name}:')
        if not 検出結果.multi_hand_landmarks:
          print('検出できませんでした')
        # cv2_imshow(cv2.flip(annotated_image, 1))
        for hand_landmarks in 検出結果.multi_hand_landmarks:
          # Print index finger tip coordinates.
          #print(
          #    f'Index finger tip coordinate: (',
          #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
          #)
          検出位置画像 = cv2.flip(画像.copy(), 1)
          mp_drawing.draw_landmarks(
            検出位置画像, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          cv2_imshow(cv2.resize(検出位置画像, dsize=None, fx=0.5, fy=0.5))
          基点 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
          指先 = [
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
          ]
          関節 = [
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
          ]
          関節までの距離 = 距離リスト(関節, 基点)
          指先までの距離 = 距離リスト(指先, 基点)
          関節までの平均距離 = np.average(np.array(関節までの距離))
          閾値 = 1.25
          スコア = 0
          if 閾値 * 関節までの平均距離 < 指先までの距離[0]:
            スコア = スコア + 1
          if 閾値 * 関節までの平均距離 < 指先までの距離[1]:
            スコア = スコア + 3
          if 閾値 * 関節までの平均距離 < 指先までの距離[2]:
            スコア = スコア + 3
          if 閾値 * 関節までの平均距離 < 指先までの距離[3]:
            スコア = スコア + 1
          if 閾値 * 関節までの平均距離 < 指先までの距離[4]:
            スコア = スコア + 1
          print('あなたの出した手は...')
          if 8 <= スコア:
            #print('パー')
            cv2_imshow(cv2.resize(手画像[0], dsize=(200,200)))
          elif 6 <= スコア:
            #print('チョキ')
            cv2_imshow(cv2.resize(手画像[1], dsize=(200,200)))
          elif スコア <= 1:
            #print('グー')
            cv2_imshow(cv2.resize(手画像[2], dsize=(200,200)))
          else:
            print('不明')
          break
  except Exception as e:
    print(e)
