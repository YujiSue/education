from .ekarte import 簡易版電子カルテ,カルテノート,診療データ
from keras.initializers import TruncatedNormal, Constant
from keras.preprocessing import image
from keras.utils import load_img, img_to_array, to_categorical
from keras.models import Sequential,model_from_json
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from IPython.display import Image, display, display_jpeg, display_png, display_pdf, clear_output
import cv2
import datetime
import glob
import requests
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

def 画像の表示(mat):
  decoded_bytes = cv2.imencode('.jpg', mat)[1].tobytes()
  display(Image(data=decoded_bytes))

###################ＡＩテスト用クラス####################
class AIテスト:
  def __init__(self):
    self.AI = ResNet50(weights='imagenet')
  
  def 予測(self, ファイルパス):
    self.画像データ = cv2.imread(ファイルパス)
    print('\n選んだ画像：')
    画像の表示(self.画像データ)
    画像 = load_img(ファイルパス, target_size=(224, 224))
    変換画像 = image.img_to_array(画像)
    変換画像 = np.expand_dims(変換画像, axis=0)
    変換画像 = preprocess_input(変換画像)
    予測 = self.AI.predict(変換画像)
    結果 = decode_predictions(予測, top=3)[0]
    要約 = {'結果':[]}
    print('\n')
    # AIが予測した、画像に映っているもの候補の中で可能性の高い結果を上から３つ表示
    print('AIが予測した画像に映っているもの')
    if len(結果) == 0:
      print('  AIが予測に失敗しました')
    else:
      print('第１候補：', 結果[0][1], '(予想確率：', '{:.2f}'.format(結果[0][2]*100), '％）')
      要約['結果'].append('名前：'+結果[0][1]+' 確率：'+'{:.2f}'.format(結果[0][2]*100))
      if len(結果) < 2:
        print('  AIが予測した候補は１つだけでした')
      else:
        print('第２候補：', 結果[1][1], '(予想確率：', '{:.2f}'.format(結果[1][2]*100), '％）')
        要約['結果'].append('名前：'+結果[1][1]+' 確率：'+'{:.2f}'.format(結果[1][2]*100))
        if len(結果) < 3:
          print('  AIが予測した候補は２つだけでした')
        else:
          print('第３候補：', 結果[2][1], '(予想確率：', '{:.2f}'.format(結果[2][2]*100), '％）')
          要約['結果'].append('名前：'+結果[2][1]+' 確率：'+'{:.2f}'.format(結果[2][2]*100))
    return 要約
#######################################################

###################ＣＶテスト用クラス####################

class 特徴抽出器:
  def __init__(self):
    self.画像データ = None
  
  def 開く(self, ファイルパス):
    self.画像データ = cv2.imread(ファイルパス)
    
  def 色抽出(self):
    高さ, 幅 = self.画像データ.shape[:2]
    比率 = 256.0/幅
    self.画像データ = cv2.resize(self.画像データ, (int(比率*幅), int(比率*高さ)))
    ラベル = ['赤', '緑', '青']
    print('元の画像')
    画像の表示(self.画像データ)
    print('\n')
    高さ, 幅 = self.画像データ.shape[:2]
    for c in range(0, 3):
      RGB画像 = self.画像データ.copy()
      print(ラベル[c], '系色のみ抽出')
      for h in range(0, 高さ):
        for w in range(0, 幅):
          for i in range(0, 3):
            if i != c:
              RGB画像[h][w][2-i] = 0
      画像の表示(RGB画像)
    return 'OK'

  def 形状認識(self, param):
    高さ, 幅 = self.画像データ.shape[:2]
    比率 = 256.0/幅
    self.画像データ = cv2.resize(self.画像データ, (int(比率*幅), int(比率*高さ)))
    print('元の画像')
    元画像 = self.画像データ.copy()
    画像の表示(元画像)
    # 直線を検出
    二値化 = cv2.cvtColor(self.画像データ, cv2.COLOR_BGR2GRAY)
    エッジ = cv2.Canny(二値化,50,300,apertureSize = 3)
    #画像の表示(エッジ)
    直線リスト = cv2.HoughLines(エッジ, rho=1, theta=np.pi/360, threshold=param['line-threshold'])
    if (直線リスト is not None) and 0 < len(直線リスト):
      # 検出した個々の直線について
      for 線 in 直線リスト:
        for rho,theta in 線:
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))
          # 検出した線を赤く引く
          cv2.line(self.画像データ,(x1,y1),(x2,y2),(0,0,255),2)
    else:
      print('直線はうまく検出できませんでした')
    
    # 円形を検出
    円リスト = cv2.HoughCircles(二値化, cv2.HOUGH_GRADIENT, dp=param['circle-dp'], minDist=100, param1=param['circle-p1'], param2=param['circle-p2'], minRadius=1)
    if (円リスト is not None) and 0 < len(円リスト[0]):
      # 検出した個々の円について
      for 円 in 円リスト[0]:
        # 検出した円を緑で縁取り
        cv2.circle(self.画像データ,(int(円[0]),int(円[1])),int(円[2]),(0,255,0),2)
    else :
      print('円はうまく検出できませんでした')
    # 表示
    画像の表示(self.画像データ)
    return 'OK'

  def 顔認識(self, param):
    高さ, 幅 = self.画像データ.shape[:2]
    比率 = 1024.0/幅
    self.画像データ = cv2.resize(self.画像データ, (int(比率*幅), int(比率*高さ)), interpolation=cv2.INTER_CUBIC)
    # 特徴分類器の読み込み
    顔検出器 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    目検出器 = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    # 計算を簡略化するためにモノクロ化
    二値化 = cv2.cvtColor(self.画像データ, cv2.COLOR_BGR2GRAY)
    # 顔を検出
    顔 = 顔検出器.detectMultiScale(二値化)
    try:
      # 検出された全員の顔について
      for (x,y,w,h) in 顔:
        if w < param['min'] or h < param['min']:
          continue
        # 検出した顔を青い四角で囲む
        cv2.rectangle(self.画像データ,(x,y),(x+w,y+h),(255,0,0),3)
        # 顔画像（グレースケール）
        顔二値 = 二値化[y:y+h, x:x+w]
        # 顔画像（カラースケール）
        顔カラー = self.画像データ[y:y+h, x:x+w]
        # 顔の中から目を検出
        目位置 = []
        目 = 目検出器.detectMultiScale(顔二値, scaleFactor=param['scale'], minNeighbors=param['dist'])
        # ある顔の中から検出された全ての目について
        for (ex,ey,ew,eh) in 目:
          目位置.append([ex,ey,ew,eh])
          # 検出した目を緑の四角で囲む
          cv2.rectangle(顔カラー,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
      print('顔：青い四角、目：緑の四角')
      高さ, 幅 = self.画像データ.shape[:2]
      比率 = 256.0/幅
      self.画像データ = cv2.resize(self.画像データ, (int(比率*幅), int(比率*高さ)))
      画像の表示(self.画像データ)
    except:
      print('上手く検出できませんでした')
    return 'OK'

#######################################################

#######################################################

# 畳み込み関数を作成
def 畳み込み(フィルタ枚数, サイズ, ストライド,  **その他の引数):
    return Conv2D(フィルタ枚数, サイズ, strides=ストライド,
                padding='same', activation='relu',
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01),
                bias_initializer=Constant(value=1),
                **その他の引数
    )
# プーリング関数を作成
def プーリング(サイズ, ストライド):
    return MaxPooling2D(pool_size=サイズ, strides=ストライド)

# 全結合関数を作成
def 全結合(ニューロン数, **その他の引数):
    return Dense(ニューロン数, 
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01),
        bias_initializer=Constant(value=1),
        **その他の引数
    )


###############　人工知能の学習用オブジェクト　##################
class AIの頭脳:
  def __init__(self):
    self.脳の構造 = Sequential()
    # 畳み込み層（１層目）
    self.脳の構造.add(畳み込み(32, 11, (4, 4), input_shape=(224, 224, 1)))
    # プーリング層
    self.脳の構造.add(プーリング((3, 3), (2, 2)))
    self.脳の構造.add(BatchNormalization())

    # 畳み込み層（２層目）
    self.脳の構造.add(畳み込み(84, 5, (1, 1)))
    # プーリング層
    self.脳の構造.add(プーリング((3, 3), (2, 2)))
    self.脳の構造.add(BatchNormalization())

    # 畳み込み層（３層目）
    self.脳の構造.add(畳み込み(128, 3, (1, 1)))
    # 畳み込み層（４層目）
    self.脳の構造.add(畳み込み(128, 3, (1, 1)))
    # 畳み込み層（５層目）
    self.脳の構造.add(畳み込み(84, 3, (1, 1)))
    # プーリング層
    self.脳の構造.add(プーリング((3, 3), (2, 2)))
    self.脳の構造.add(BatchNormalization())

    # 全結合層
    # １次元のニューロン列へ
    self.脳の構造.add(Flatten())
    # 全結合x２
    self.脳の構造.add(全結合(4096))
    self.脳の構造.add(Dropout(0.5))
    self.脳の構造.add(全結合(4096))
    self.脳の構造.add(Dropout(0.5))

    # 出力層
    # 状態は健常か心拡大かの２通り
    self.脳の構造.add(全結合(2, activation='softmax'))
    
    # 脳の構造を最終確定
    self.脳の構造.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    # 学習過程を記録しておく
    # どんな風に賢くなったかが見れる
    self.学習の履歴 = None

  # 脳の構造をあとで確認したくなったとき用
  # 絵面を想像すると微妙だった...
  # 関数名はちゃんと考えてつけましょうという教訓
  def 頭の中身(self):
    self.脳の構造.summary()
    
  # 実際の学習を行う関数
  # 今回は教師あり(健常か心拡大か過去に医師が判定した結果がある)学習
  def 勉強(self, 画像, 教師, 世代数=60):
    self.学習の履歴 = self.脳の構造.fit(画像, 教師, batch_size=32, epochs=世代数)
  
  # 未知の患者のデータに対して、健常かそうでないか予測する関数
  def 予測(self, テスト画像):
    return self.脳の構造.predict(テスト画像, verbose=0)

  # 記録しておいた学習過程をグラフで表示
  def 進捗の確認(self):
    plt.plot(self.学習の履歴.history['accuracy'],"o-",label="accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


###############　人工知能オブジェクト　##################
class 画像診断AI:
  # 初期状態の設定
  def __init__(self):
    self.脳 = AIの頭脳()
  
  # 学習関数の作成
  def 学習(self, 画像, 診断):
    self.脳.勉強(画像, 診断)

  # 過去の学習済データを呼び出す
  def 思い出す(self, 記憶):
    self.脳.脳の構造 = model_from_json(open(記憶['モデル'],"r").read())
    self.脳.脳の構造.load_weights(記憶['結果'])

  # 診断関数の作成
  def 判断(self, 診療データ):
    # 診療データから画像名を取得
    検査データ = 診療データ.診療結果['X線写真']
    # X線画像を読み込み、人工知能に渡せるように変換
    検査結果画像 = []
    画像データ = load_img(検査データ, color_mode = "grayscale", target_size=(224, 224))
    検査結果画像.append(img_to_array(画像データ))
    検査結果画像 = np.array(検査結果画像)
    検査結果画像 = 検査結果画像.astype('float32')
    検査結果画像 = 検査結果画像 / 255.0
    # 人工知能の予測を取得
    予測結果 = self.脳.予測(検査結果画像)[0]
    # 予測結果は確率の配列で出てくる
    
    if (予測結果[0] < 予測結果[1]):
      診療データ.所見の記入('心拡大')
    else: 
      診療データ.所見の記入('正常')
