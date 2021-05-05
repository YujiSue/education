# -*- coding: utf-8 -*-
from .test import Test
from .ekarte import 簡易版電子カルテ
from .ekarte import カルテページ

def sample():
  sample_karte = 簡易版電子カルテ()
  pages = []
  for i in range(0, 10):
    pages.append(カルテページ())
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
  
  for p in pages:
    sample_karte.ページの追加(p)
  return sample_karte
  

def printTest():
  print('Test')
