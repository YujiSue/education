# -*- coding: utf-8 -*-
import base64
import os
import json
import datetime
import numpy as np
import requests
import cv2
from IPython.display import HTML, Image, display

###############　サンプルデータ　##################
class サンプルデータ:
  def __init__(self):
    current = os.path.dirname(os.path.abspath(__file__))
    self.データ = json.load(open(os.path.join(current, 'sample.json')))
    res = requests.get('https://github.com/YujiSue/education/blob/main/ekarte/sample.png?raw=true')
    with open('sample.png', 'wb') as f:
      f.write(res.content)

###############　簡易版電子カルテオブジェクト　##################
class 簡易版電子カルテシステム:
  def __init__(self):
    self.患者カルテリスト = []

  def ページの追加(self, ページ):
    self.患者カルテリスト.append(ページ)

  def 開く(self, ページ番号):
    if ページ番号 <= len(self.患者カルテリスト):
      return self.患者カルテリスト[ページ番号-1]
    else:
      print(ページ番号,'番目のノートは存在しません')

  def 表示(self, ページ番号, 展開 = False):
    if ページ番号 <= len(self.患者カルテリスト):
      self.患者カルテリスト[ページ番号-1].表示(展開)
    else:
      print(ページ番号,'番目のノートは存在しません')

  def 読み込み(self, データ):
    for 情報 in データ:
      ページ = カルテページ()
      ページ.セット(情報)
      self.ページの追加(ページ)
  
###############　カルテページのオブジェクト　##################
class カルテページ:
  def __init__(self):
    self.患者ID = ''
    self.患者氏名 = ''
    self.患者生年月日 = ''
    self.患者性別 = ''
    self.診療データリスト = []

  def 記入(self, ID, 氏名, 生年月日, 性別):
    self.患者ID = ID
    self.患者氏名 = 氏名
    self.患者生年月日 = 生年月日
    self.患者性別 = 性別

  def 年齢(self):
    現在 = datetime.datetime.now()
    生年月日 = self.患者生年月日.split('-')
    生年 = int(生年月日[0])
    生月 = int(生年月日[1])
    return 現在.year - 生年 - (0 if 生月 < 現在.month else 1)

  def 診療データ追加(self, データ):
    self.診療データリスト.append(データ)

  def 要約(self):
    テキスト = ''
    テキスト = 'ID:'+self.患者ID+'<br/>\n氏名:'+self.患者氏名+'<br/>\n年齢:'+str(self.年齢())+'歳<br/>\n<br/>\n'
    テキスト = テキスト+'診療データ数:'+str(len(self.診療データリスト))+'<br/>\n'
    if (0 < len(self.診療データリスト)):
      for i in range(0, len(self.診療データリスト)):
        テキスト = テキスト+'> No.'+str(i+1)+':<br/>\n'+self.診療データリスト[i].データの取得()+'<br/>\n'
    return テキスト

  def セット(self, データ):
    self.患者ID = データ['id']
    self.患者氏名 = データ['name']
    self.患者生年月日 = データ['birth']
    self.患者性別 = データ['sex']
    for 情報 in データ['diagnos']:
      サブデータ = 診療データ()
      サブデータ.セット(情報)
      self.診療データ追加(サブデータ)

  def 表示(self, expand = False):
    # 患者の基本情報
    html = f'''
    <div style="display:flex; flex-direction:row; justify-content:stretch; background-color: oldlace; padding: 10px;">
      <div style="padding: 5px;">
      <div style="font-size:22px; font-weight: bold; margin-bottom: 10px;">基本情報</div>
      <table border=0 cellspacing="0" style="border:2.5px solid black; font-size: 18px;">
        <tr><td style="border:1px solid black; padding:10px;">ID</td><td style="border:1px solid black; padding:10px">{self.患者ID}</td></tr>
        <tr><td style="border:1px solid black; padding:10px">氏名</td><td style="border:1px solid black; padding:10px">{self.患者氏名}</td></tr>
        <tr><td style="border:1px solid black; padding:10px">生年月日</td><td style="border:1px solid black; padding:10px">{self.患者生年月日} ({self.年齢()} 歳)</td></tr>
        <tr><td style="border:1px solid black; padding:10px">性別</td><td style="border:1px solid black; padding:10px">{self.患者性別}</td></tr>
      </table>
      </div>
      <div style="border-right:0.5px dotted black; margin-right:10px; margin-left:10px;"></div>
      <div style="flex: auto; padding: 5px;">
        <div style="font-size:22px; font-weight: bold; margin-bottom: 10px;">診療記録</div>
        %_DIAGNOS_RECORD_%
      </div>
    </div>
    '''
    # 診療情報
    content = ''
    for (インデックス, 情報) in enumerate(self.診療データリスト):
      content += f'''
      <details style="border: 1px solid black; padding: 10px; margin: 5px;" {'open' if expand else ''}>
        <summary style="font-size:18px; margin-bottom: 10px;">診療データ No. {インデックス + 1}</summary>
        <div style="display:flex; justify-content:stretch; flex-direction:row; margin-bottom: 20px; font-size: 18px; border-bottom: 1px solid black;">
          <div>診療： {情報.診療名}</div>
          <div style="flex:auto;"></div>
          <div>診療日： {情報.診療日.strftime('%Y年%m月%d日')}</div>
        </div>
      '''
      if 情報.データタイプ == '画像':
        for 項目 in 情報.診療項目:
          画像データ = cv2.imread(情報.診療結果[項目])
          画像データ = cv2.resize(画像データ, (256, 256))
          画像データ = cv2.cvtColor(画像データ, cv2.COLOR_BGR2RGB)
          (結果, 画像データ) = cv2.imencode('.jpg', 画像データ)
          画像データ = base64.b64encode(画像データ)
          content += f'''<div>
          <h2>{項目}</h2>
          <img src="data:image/jpeg;base64,{画像データ.decode('utf-8')}">
          </div>
          '''
      else:
        content += f'''<table border=0 cellspacing="0" style="border:1px solid black; font-size:18px;">
        <tr><th style="border:1px solid black; padding:10px;">診療項目</th><th style="border:1px solid black; padding:10px;">結果</th></tr>
        '''
        for 項目 in 情報.診療項目:
          content += f'''
          <tr><td style="border:1px solid black; padding:10px;">{項目}</td><td style="border:1px solid black; padding:10px;">{情報.診療結果[項目]}</td></tr>
          '''
        content += f'''</table>'''
      content += f'''
      <div style="margin-top: 10px;">
      <div style="font-size:22px; font-weight: bold; margin-bottom: 10px;">所見：</div>
      <div style="display:inline-block; border: 0.5px solid black; border-radius:5px; width: 90%; height: 24px; padding: 0.5em; font-size: 16px;">{情報.医師の所見}</div></div>
      </details>
      '''
    # 診療情報を統合
    html = html.replace('%_DIAGNOS_RECORD_%', content)
    # 表示
    display(HTML(html))

###############　診療データオブジェクト　##################
class 診療データ:
  def __init__(self):
    self.診療名 = ''
    self.診療日 = datetime.date.today()
    self.診療項目 = []
    self.データタイプ = ''
    self.診療結果 = {}
    self.医師の所見 = ''
  
  def 診療情報の記入(self, 内容, タイプ='表'):
    self.診療名 = 内容['名称']
    self.診療項目 = 内容['項目']
    self.データタイプ = タイプ

  def 診療結果の記入(self, 項目, 結果):
    self.診療結果[項目] = 結果

  def 所見の記入(self, 所見):
    self.医師の所見 = 所見
  
  def データの取得(self):
    内容 = ''
    for 項目 in self.診療項目:
      内容 = 内容 + 項目 + ':' + str(self.診療結果[項目]) + '<br/>\n'
    内容 = 内容 + '所見：' + self.医師の所見
    return 内容

  def セット(self, データ):
    self.診療名 = データ['name']
    self.データタイプ = データ['type']
    for 情報 in データ['list']:
      self.診療項目.append(情報['name'])
      self.診療結果[情報['name']] = 情報['value']
    self.医師の所見 = データ['note']
    