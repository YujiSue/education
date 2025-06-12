import os
import math
import datetime
import random

import base64
from base64 import b64decode

import glob
import requests
from matplotlib import ticker
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2

from keras.preprocessing import image
from keras.utils import load_img, img_to_array, to_categorical
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50

from IPython.display import Image, HTML, display, Javascript, clear_output
from google.colab import output

################### Base64エンコーダ ####################

def 埋め込み画像エンコーダ(画像データ, サイズ=(-1, -1), 形式 = 'jpeg'):
  ピクセル = 画像データ.shape[:2]
  画像 = 画像データ
  if -1 < サイズ[0]:
    if -1 < サイズ[1]:
      画像 = cv2.resize(画像データ, (サイズ[0], サイズ[1]))
    else:
      画像 = cv2.resize(画像データ, (サイズ[0], int(サイズ[0]*ピクセル[0]/ピクセル[1])))
  elif -1 < サイズ[1]:
    画像 = cv2.resize(画像データ, (int(サイズ[1]*ピクセル[1]/ピクセル[0]), サイズ[1]))
  画像 = cv2.cvtColor(画像, cv2.COLOR_BGR2RGB)
  (コード, データ) = cv2.imencode(('.jpg' if 形式 == 'jpeg' else '.png'), 画像)
  return f"data:image/{形式};base64,{base64.b64encode(データ).decode('utf-8')}"

################### 画像ダウンローダー ####################

class 画像ダウンローダー:
  def __init__(self, name):
    self.画像名 = name
    current = os.path.dirname(os.path.abspath(__file__))
    self.画像データリスト = pd.read_csv(os.path.join(current, 'test-images-with-rotation.csv'))
    self.画像数 = len(self.画像データリスト)
    self.選択した画像のURL = ''

  def selectImage(self):
    print(self.選択した画像のURL)
    os.system(f'curl -L -s -o "{self.画像名}.jpg" "{self.選択した画像のURL}"')
    print(f"{self.画像名}.jpgとしてダウンロード")

  def changeImage(self):
    clear_output()
    self.選択した画像の番号 = random.randrange(self.画像数)
    self.選択した画像のURL = self.画像データリスト['OriginalURL'][self.選択した画像の番号]
    display(Image(url=self.選択した画像のURL, width=256))
    display(HTML(f'''
      <button id="change">別の画像にする</button>
      <span style="margin:20px">
      <button id="select">この画像にする</button>
      <div style="margin-bottom:10px;"></div>
      <script>
        document.querySelector("#change").onclick = function(e) {{
          google.colab.kernel.invokeFunction("notebook.changeImage", [], {{}});
        }};
        document.querySelector("#select").onclick = function(e) {{
          google.colab.kernel.invokeFunction("notebook.selectImage", [], {{}});
        }};
      </script>'''))

################### 画像に映ったもの判定用クラス ####################

class 画像判定器:
  # 既成AI (ResNet)を取得
  def __init__(self):
    self.AI = ResNet50(weights='imagenet')

  # 画像に映っているものを予測
  def 予測(self, 画像ファイル):
    self.画像データ = cv2.imread(画像ファイル)
    self.画像データ = cv2.cvtColor(self.画像データ, cv2.COLOR_BGR2RGB)
    self.画像データ = 埋め込み画像エンコーダ(self.画像データ)
    self.予測結果 = []
    # 画像読み込み
    画像 = load_img(画像ファイル, target_size=(224, 224))
    # AIが理解できるように変換
    変換画像 = image.img_to_array(画像)
    変換画像 = np.expand_dims(変換画像, axis=0)
    変換画像 = preprocess_input(変換画像)
    # AIの予測
    予測 = self.AI.predict(変換画像)
    # (あれば)予測結果TOP3の取得
    結果 = decode_predictions(予測, top=3)[0]
    if len(結果) == 0:
      # 予測失敗
      return
    else:
      self.予測結果.append({'label': 結果[0][1], 'prob': '{:.2f}'.format(結果[0][2]*100)})
      # 第２候補がある場合
      if 1 < len(結果) :
        self.予測結果.append({'label': 結果[1][1], 'prob': '{:.2f}'.format(結果[1][2]*100)})
      # 第３候補がある場合
      if 2 < len(結果) :
        self.予測結果.append({'label': 結果[2][1], 'prob': '{:.2f}'.format(結果[2][2]*100)})


  def 結果を表示(self):
    html = f'''
    <div><h3>元画像：</h3>
    <img src="{self.画像データ}" width="256">
    </div>
    <div><h3>予測結果：</h3>
    '''
    if len(self.予測結果) == 0:
      html += f'''
      <p style="font-size: 18px; color:red">予測できませんでした</p>
      </div>
      '''
    else:
      html += f'''
      <table>
      <tr style="font-size: 18px;"><th></th><th style="padding: 6px;">映っていたもの</th><th style="padding: 6px;">確率</th></tr>
      '''
      for 番号, 候補 in enumerate(self.予測結果):
        html += f'''<tr style="font-size: 18px;"><td style="padding: 6px;">第{番号+1}候補</td><td style="padding: 6px;">{候補['label']}</td><td style="padding: 6px;">{候補['prob']}</td></tr>'''
      html += f'''
      </table>
      </div>
      '''
    # 表示
    display(HTML(html))

###################ＣＶテスト用クラス####################
 
class 特徴抽出器:
  def __init__(self):
    self.画像ファイル = None
    self.画像データ = None
    self.抽出結果 = None

  def 開く(self, ファイル):
    self.画像ファイル = ファイル
    self.画像データ = cv2.imread(ファイル)
    self.画像データ = cv2.cvtColor(self.画像データ, cv2.COLOR_BGR2RGB)
    self.画像データ = 埋め込み画像エンコーダ(self.画像データ, サイズ=(256,-1))

  def 色情報の抽出(self):
    self.抽出結果 = []
    画像 = cv2.imread(self.画像ファイル)
    画像 = cv2.cvtColor(画像, cv2.COLOR_BGR2RGB)
    スケール = 画像.shape[:2]
    画像 = cv2.resize(画像, (256, int(256.0*スケール[0]/スケール[1])))
    スケール = 画像.shape[:2]
    # RGB順に抽出
    ラベル = ['赤', '緑', '青']
    for c in range(0, 3):
      単色画像 = np.zeros((スケール[0], スケール[1], 3), dtype=np.uint8)
      for h in range(0, スケール[0]):
        for w in range(0, スケール[1]):
          単色画像[h][w][c] = 画像[h][w][c]
      self.抽出結果.append({
        'annotation': f"{ラベル[c]}色成分",
        'image' : 埋め込み画像エンコーダ(単色画像, サイズ=(-1,-1))
      })

  def 形状の認識(self, param):
    self.抽出結果 = []
    画像 = cv2.imread(self.画像ファイル)
    画像 = cv2.cvtColor(画像, cv2.COLOR_BGR2RGB)
    スケール = 画像.shape[:2]
    画像 = cv2.resize(画像, (256, int(256.0*スケール[0]/スケール[1])))
    スケール = 画像.shape[:2]
    # 二値化とエッジ検出
    二値化 = cv2.cvtColor(画像, cv2.COLOR_BGR2GRAY)
    エッジ = cv2.Canny(二値化, int(param['bthreshold']/2), param['bthreshold'], apertureSize = 3)
    # 直線を検出
    直線リスト = cv2.HoughLinesP(エッジ, rho=1, theta=np.pi/180, threshold=param['lthreshold'], minLineLength=param['minl'],maxLineGap=param['ming'])
    if 直線リスト is not None and 0 < len(直線リスト):
      # 検出した個々の直線について
      for 線 in 直線リスト:
        x1, y1, x2, y2 = 線[0]
        # 検出した線を赤く引く
        cv2.line(画像, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 円形を検出
    円リスト = cv2.HoughCircles(二値化, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=param['bthreshold'], param2=param['cthreshold'], minRadius=param['minr'])
    if 円リスト is not None and 0 < len(円リスト[0]):
      # 検出した個々の円について
      for 円 in 円リスト[0]:
        # 検出した円を緑で縁取り
        cv2.circle(画像,(int(円[0]),int(円[1])),int(円[2]),(127,255,0),3)

    self.抽出結果.append({
        'annotation': '直線的な部分：青、円形の図形：緑',
        'image': 埋め込み画像エンコーダ(画像, サイズ=(-1,-1))
      })


  def 顔認識(self, param):
    self.抽出結果 = []
    画像 = cv2.imread(self.画像ファイル)
    画像 = cv2.cvtColor(画像, cv2.COLOR_BGR2RGB)
    スケール = 画像.shape[:2]
    画像 = cv2.resize(画像, (512, int(512.0*スケール[0]/スケール[1])))
    # 計算を簡略化するためにモノクロ化
    二値化 = cv2.cvtColor(画像, cv2.COLOR_BGR2GRAY)

    # 特徴分類器の読み込み
    current = os.path.dirname(os.path.abspath(__file__))
    顔検出器 = cv2.CascadeClassifier(os.path.join(current, 'haarcascade_frontalface_default.xml'))
    目検出器 = cv2.CascadeClassifier(os.path.join(current, 'haarcascade_eye.xml'))

    # 顔を検出
    顔 = 顔検出器.detectMultiScale(二値化, minSize=param['size'])

    for idx,(x,y,w,h) in enumerate(顔):
        # 検出した顔を青い四角で囲む
        cv2.rectangle(画像,(x,y),(x+w,y+h),(0,255,0),3)
        # 顔画像（グレースケール）
        顔二値 = 二値化[y:y+h, x:x+w]
        # 顔画像（カラースケール）
        顔カラー = 画像[y:y+h, x:x+w]
        # 顔の中から目を検出
        目位置 = []
        目 = 目検出器.detectMultiScale(顔二値, scaleFactor=param['scale'], minNeighbors=param['dist'])
        # ある顔の中から検出された全ての目について
        for (ex,ey,ew,eh) in 目:
          目位置.append([ex,ey,ew,eh])
          # 検出した目を緑の四角で囲む
          cv2.ellipse(顔カラー,(ex+int(ew/2),ey+int(eh/2)),(ew,eh),0,0,360,(0,0,255),3)

    self.抽出結果.append({
      'annotation': f"顔：緑の四角、目：青の楕円",
      'image' : 埋め込み画像エンコーダ(画像, サイズ=(256,-1))
    })

  def 結果の表示(self):
    html = f'''
    <div><h3>元画像</h3><img src="{self.画像データ}"></div>
    <div><h3>抽出された特徴</h3>
      <div style="display:flex; flex-direction:row;">{'<div style="flex:auto;"><span style="color:red">抽出できませんでした</span></div>' if len(self.抽出結果) == 0 else ''}
    '''
    for 結果 in self.抽出結果:
      html += f'''
      <div style="flex:auto;"><p style="color:blue">{結果['annotation']}</p><img src="{結果['image']}"></div>
      '''
    html += '</div></div>'
    display(HTML(html))

#######################################################
