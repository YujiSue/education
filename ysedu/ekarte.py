# -*- coding: utf-8 -*-
class 簡易版電子カルテ:
  def __init__(self):
    患者リスト = []
    
  def カルテの追加(self, ページ):
    患者リスト.append(ページ)
    
  def 開く(self, ページ番号):
    return 患者リスト[ページ番号]
  
class カルテページ:
  def __init__(self):
    self.患者ID = ''
    self.患者氏名 = ''
    
  def 記入(self, ID, 氏名, 生年月日):
    self.患者ID = ID
    
    
