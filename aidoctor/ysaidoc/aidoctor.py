# -*- coding: utf-8 -*-
import glob
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from google.colab.output import eval_js

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


#######################################################
class 画像セレクタ:
  def __init__(self, dir):
    self.画像名 = f"test.png"
    self.画像データリスト = glob.glob(f"{dir}/*.png")
    self.画像数 = len(self.画像データリスト)
    self.乱数範囲 = random.randrange(self.画像数)
    self.選択した画像の番号 = 0
    self.選択した画像 = ''

  def selectImage(self):
    print('テスト画像：', self.選択した画像)
    os.system(f'mv "{self.選択した画像}" "{self.画像名}"')

  def changeImage(self):
    clear_output()
    self.選択した画像の番号 = self.乱数範囲.integers(self.画像数)
    self.選択した画像 = self.画像データリスト[self.選択した画像の番号]
    display(Image(self.選択した画像, width=256))
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

###############　人工知能の学習用オブジェクト　##################
class ニューラルネット:
  def __init__(self):
    # AIの構造を保持する変数
    self.構造 = Sequential()
    # 学習過程の記録用変数
    self.記録 = None
    # AIを構築
    self.画像サイズ = None
    
  # 使用する画像のサイズを設定
  def 画像サイズの設定(self, サイズ):
    self.画像サイズ = サイズ

  # AIを最終構築
  def 構築(self):
    # AIの構造を最終確定
    self.構造.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

  # AIの構造を確認したくなったとき用
  def 構造を表示(self):
    self.構造.summary()
    
  # 実際の学習を行う関数
  # 今回は教師あり(健常か心拡大か過去に医師が判定した結果がある)学習
  def 機械学習(self, 画像, 教師, 反復数):
    self.記録 = self.構造.fit(画像, 教師, batch_size=32, epochs=反復数)
  
  # 未知の患者のデータに対して、健常かそうでないか予測する関数
  def 予測(self, テスト画像):
    return self.構造.predict(テスト画像, verbose=0)

  # 記録しておいた学習過程をグラフで表示
  def 進捗の確認(self):
    plt.plot(self.記録.history['accuracy'],"o-",label="accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

#########################################################################

  def 畳み込み層の追加(self, カーネル数, カーネルサイズ, 移動量, 活性化関数, 入力画像サイズ = None, **その他の引数):
    self.構造.add(Conv2D(カーネル数, カーネルサイズ, strides=移動量,
                activation=活性化関数, padding='same', 
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01),
                bias_initializer=Constant(value=1),
                input_shape = 入力画像サイズ,
                **その他の引数))

  def プーリング層の追加(self, プールサイズ, 移動量):
    self.構造.add(MaxPooling2D(pool_size=プールサイズ, strides=移動量))
    self.構造.add(BatchNormalization())
  
  # １次元のノード列へ
  def 情報の統合(self):
    self.構造.add(Flatten())

  def 全結合層の追加(self, ノード数, 活性化関数 = None, drop=None, **その他の引数):
    self.構造.add(Dense(ノード数, 
        activation=活性化関数,
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01),
        bias_initializer=Constant(value=1),
        **その他の引数))
    if drop:
      self.構造.add(Dropout(drop))

###############　人工知能オブジェクト　##################
class AI医師:
  # 初期状態の設定
  def __init__(self, 診断する画像サイズ = (256, 256, 1)):
    self.頭脳 = ニューラルネット()
    self.頭脳.画像サイズの設定(診断する画像サイズ)
    
  def 標準構成(self):
    self.頭脳.畳み込み層の追加(32, 12, (4,4), "relu", 入力画像サイズ = self.頭脳.画像サイズ)
    self.頭脳.プーリング層の追加((3,3), (2,2))
    self.頭脳.畳み込み層の追加(64, 5, (1,1), "relu")
    self.頭脳.プーリング層の追加((3,3), (2,2))
    self.頭脳.畳み込み層の追加(128, 3, (1,1), "relu")
    self.頭脳.畳み込み層の追加(128, 3, (1,1), "relu")
    self.頭脳.畳み込み層の追加(64, 3, (1,1), "relu")
    self.頭脳.プーリング層の追加((3,3), (2,2))
    self.頭脳.情報の統合()
    self.頭脳.全結合層の追加(4096, drop=0.5)
    self.頭脳.全結合層の追加(4096, drop=0.5)
    self.頭脳.全結合層の追加(2, 'softmax')
    self.頭脳.構築()
  
  # 学習関数の作成
  def 学習(self, 画像, 教師, 反復数):
    self.頭脳.機械学習(画像, 教師, 反復数)

  # 学習済データを残す
  def メモ(self, ファイル):
    open(f"{ファイル}.model.json","w").write(self.頭脳.構造.to_json())
    self.頭脳.構造.save_weights(f"{ファイル}.param.weights.h5")

  # 過去の学習済データを呼び出す
  def 思い出す(self, ファイル):
    self.頭脳.構造 = model_from_json(open(f"{ファイル}.model.json").read())
    self.頭脳.構造.load_weights(f"{ファイル}.param.weights.h5")

  # 診断関数の作成
  def 診断(self, 検査画像):
    # 検査画像を読み込み、人工知能に渡せるように変換
    検査結果画像 = []
    画像データ = load_img(検査画像, color_mode = "grayscale", target_size=self.頭脳.画像サイズ[:2])
    検査結果画像.append(img_to_array(画像データ))
    検査結果画像 = np.array(検査結果画像)
    検査結果画像 = 検査結果画像.astype('float32')
    検査結果画像 = 検査結果画像 / 255.0
    # 人工知能の予測を取得
    予測結果 = self.頭脳.予測(検査結果画像)[0]
    # 正常(0)か心拡大(1)か、”より活性化している”ノードを採用
    if (予測結果[0] < 予測結果[1]):
      return '心拡大'
    else: 
      return '正常'

