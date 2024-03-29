# -*- coding: utf-8 -*-
#from .test import Test
from IPython.display import Image, display
import cv2
from .ekarte import 簡易版電子カルテ,カルテノート,診療データ
from .ai import AIテスト,特徴抽出器,AIの頭脳,画像診断AI
from .janken import じゃんけん判定
import copy

def 画像の表示(mat):
  decoded_bytes = cv2.imencode('.jpg', mat)[1].tobytes()
  display(Image(data=decoded_bytes))

def sample():
  sample_karte = 簡易版電子カルテ()
  pages = []
  for i in range(0, 10):
    pages.append(カルテノート())
  pages[0].記入(110001, '山田太郎', '2000-04-01', '男')
  pages[1].記入(110002, '鈴木次郎', '1990-05-04', '男')
  pages[2].記入(110003, '佐藤花子', '1980-06-07', '女')
  pages[3].記入(110004, '高橋三郎', '1970-07-10', '男')
  pages[4].記入(110005, '田中牡丹', '1960-08-13', '女')
  pages[5].記入(110006, '中村紅葉', '1950-09-16', '女')
  pages[6].記入(110007, '小林四郎', '2005-10-19', '男')
  pages[7].記入(110008, '加藤萩', '2010-01-22', '女')
  pages[8].記入(110009, '渡辺五郎', '2015-02-25', '男')
  pages[9].記入(110010, '伊藤陸', '2020-03-28', '男')
  
  dat01 = 診療データ()
  dat01.診療情報の記入({'名称':'定期健康診断','項目': ['身長 (cm)', '体重 (kg)', '視力(両眼)', '血圧(mmHg)']}, '表')
  dat02 = 診療データ()
  dat02.診療情報の記入({'名称':'血液検査','項目': ['赤血球数', '白血球数', 'γ-GTP ', '血糖　　']}, '表')
  dat03 = 診療データ()
  dat03.診療情報の記入({'名称':'新型コロナ検査','項目': ['体温', 'せき', '味覚', 'PCR ']}, '表')
  
  dat1 = copy.deepcopy(dat01)
  dat1.診療結果の記入('身長 (cm)', 165)
  dat1.診療結果の記入('体重 (kg)', 60)
  dat1.診療結果の記入('視力(両眼)', 1.0)
  dat1.診療結果の記入('血圧(mmHg)', '120/75')
  dat1.所見の記入('問題なし')
  pages[0].診療データ追加(dat1)
  dat2 = copy.deepcopy(dat01)
  dat2.診療結果の記入('身長 (cm)', 170)
  dat2.診療結果の記入('体重 (kg)', 70)
  dat2.診療結果の記入('視力(両眼)', 0.8)
  dat2.診療結果の記入('血圧(mmHg)', '145/95')
  dat2.所見の記入('高血圧')
  pages[1].診療データ追加(dat2)
  dat22 = copy.deepcopy(dat02)
  dat22.診療結果の記入('赤血球数', '400万')
  dat22.診療結果の記入('白血球数', 5000)
  dat22.診療結果の記入('γ-GTP ', 25)
  dat22.診療結果の記入('血糖　　', 120)
  dat22.所見の記入('高血糖')
  pages[1].診療データ追加(dat22)
  dat3 = copy.deepcopy(dat01)
  dat3.診療結果の記入('身長 (cm)', 175)
  dat3.診療結果の記入('体重 (kg)', 65)
  dat3.診療結果の記入('視力(両眼)', 0.5)
  dat3.診療結果の記入('血圧(mmHg)', '110/70')
  dat3.所見の記入('問題なし')
  pages[2].診療データ追加(dat3)

  dat10 = copy.deepcopy(dat03)
  dat10.診療結果の記入('体温', 36.5)
  dat10.診療結果の記入('せき', 'あり')
  dat10.診療結果の記入('味覚', '正常')
  dat10.診療結果の記入('PCR ', '未検出')
  dat10.所見の記入('新型コロナではない')
  pages[9].診療データ追加(dat10)

  for p in pages:
    sample_karte.ノートの追加(p)
  return sample_karte
  

#def printTest():
#  print('Test')
