from keras.initializers import TruncatedNormal, Constant
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib import ticker
import glob
import numpy as np
import datetime

# もともとkerasに用意されていた関数を利用して畳み込み関数を作成
def 畳み込み(フィルタ枚数, サイズ, ストライド,  **その他の引数):
    return Conv2D(フィルタ枚数, サイズ, strides=ストライド,
                padding='same', activation='relu',
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01),
                bias_initializer=Constant(value=1),
                **その他の引数
    )
# もともとkerasに用意されていた関数を利用してプーリング関数を作成
def プーリング(サイズ, ストライド):
    return MaxPooling2D(pool_size=サイズ, strides=ストライド)

# もともとkerasに用意されていた関数を利用して全結合関数を作成
def 全結合(ニューロン数, **その他の引数):
    return Dense(ニューロン数, 
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01),
        bias_initializer=Constant(value=1),
        **その他の引数
    )


###############　人工知能の学習用オブジェクト　##################
class AIの脳みそ:
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
    self.脳の構造.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

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
  def 勉強(self, 画像, 教師):
    self.学習の履歴 = self.脳の構造.fit(学習用画像, 教師, batch_size=32, epochs=60)
  
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
    self.脳 = AIの脳みそ()
  
  # 学習関数の作成
  def 学習(self, 画像, 診断):
    self.脳.勉強(画像, 診断)

  # 判断関数の作成
  def 判断(self, 診療データ):
    # 診療データから画像名を取得
    検査データ = 診療データ.診療結果['X線写真']
    # X線画像を読み込み、人工知能に渡せるように変換
    検査結果画像 = []
    画像データ = image.load_img(検査データ, color_mode = "grayscale", target_size=(224, 224))
    検査結果画像.append(image.img_to_array(画像データ))
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