# -*- coding: utf-8 -*-
###############　簡易版電子カルテオブジェクト　##################
class 簡易版電子カルテ:
  def __init__(self):
    self.患者リスト = []
    
  def ページの追加(self, ページ):
    self.患者リスト.append(ページ)
    
  def 開く(self, ページ番号):
    self.患者リスト[ページ番号-1].表示() if ページ番号 <= len(self.患者リスト) else print(ページ番号,'ページは存在しません')
  
###############　カルテページオブジェクト　##################
class カルテページ:
  def __init__(self):
    self.患者ID = 0
    self.患者氏名 = ''
    self.患者生年月日 = ''
    self.性別 = ''
    self.診療データリスト = []
    
  def 記入(self, ID, 氏名, 生年月日, 性別):
    self.患者ID = ID
    self.患者氏名 = 氏名
    self.患者生年月日 = 生年月日
    self.患者性別 = 性別
    
  def 年齢(self):
    生年月日 = self.患者生年月日.split('-')
    生年 = int(生年月日[0])
    生月 = int(生年月日[1])
    return 2021 - 生年 + (1 if 生月 < 5 else 0)
  
  def 診療データ追加(self, データ):
    self.診療データリスト.append(データ)

  def 表示(self):
    #患者の基本情報
    print('============================================================')
    print('ID:　',self.患者ID)
    print('氏名:　',self.患者氏名)
    print('生年月日:　',self.患者生年月日, '(', self.年齢(), '歳)')
    print('性別:　',self.患者性別)
    print('============================================================')
    if (0 < len(self.診療データリスト)):
      print('')
      
###############　診療データオブジェクト　##################
class 診療データ:
  def __init__(self):
    self.診療名 = ''
    self.診療日 = datetime.date.today()
    self.診療項目 = []
    self.診療結果 = {}
    self.医師の所見 = ''
  
  def 診療情報の記入(self, 内容):
    self.診療名 = 内容['名称']
    self.診療項目 = 内容['項目']

  def 診療結果の記入(self, 項目, 結果):
    self.診療結果[項目] = 結果

  def 所見の記入(self, 所見):
    self.医師の所見 = 所見

  def データのまとめ(self):
    データ = '－－－－－－－－－－－－－－－－－－－－－\n'
    データ += '診療：'+self.診療名+'　　診療日：'+self.診療日.strftime('%y年%m月%d日')+'\n'
    データ += '－－－－－－－－－－－－－－－－－－－－－\n'
    データ += '　診療項目　 結果\n'
    for 項目名 in self.診療項目:
      if (項目名 == 'X線写真'):
        self.画像あり = True
        self.画像 = cv2.imread(self.診療結果[項目名])
        self.画像 = cv2.resize(self.画像, (256, 256))
      else:
        データ += '　　'+項目名+'　　'+self.診療結果[項目名]+'\n'
    データ += '\n所見：'+self.医師の所見+'\n'
    return データ


    
