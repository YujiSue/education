# -*- coding: utf-8 -*-
import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4, portrait
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import mm
from reportlab.lib import colors

def init(pdf_canvas):
  pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
  width, height = A4
    
def printHead(pdf_canvas, data):
  lid = data['lid']
  sid = data['sid']
  if len(lid) < 8:
    lid_ = ''
    k = 8 - len(lid)
    for i in (0, k):
      lid_ = lid_ + ' '
    lid = lid_ + lid

  if len(sid) < 8:
    sid_ = ''
    k = 8 - len(sid)
    for i in (0, k):
      sid_ = sid_ + ' '
    sid = sid_ + sid

  data1 = [['講義ID',lid[0:1],lid[1:2],lid[2:3],lid[3:4],lid[4:5],lid[5:6],lid[6:7],lid[7:8]]]
  table1 = Table(data1)
  table1.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'HeiseiKakuGo-W5', 8),
            ('BOX', (0, 0), (-1, 0), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, 0), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, 0), colors.black),
            ('TEXTCOLOR',(0,0),(0,0),colors.white),
        ]))
  table1.wrapOn(pdf_canvas, 15*mm, 280*mm)
  table1.drawOn(pdf_canvas, 15*mm, 280*mm)
  data2 = [['学生ID',sid[0:1],sid[1:2],sid[2:3],sid[3:4],sid[4:5],sid[5:6],sid[6:7],sid[7:8]]]
  table2 = Table(data2)
  table2.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'HeiseiKakuGo-W5', 8),
            ('BOX', (0, 0), (-1, 0), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, 0), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, 0), colors.black),
            ('TEXTCOLOR',(0,0),(0,0),colors.white),
        ]))
  table2.wrapOn(pdf_canvas, 125*mm, 280*mm)
  table2.drawOn(pdf_canvas, 125*mm, 280*mm)
  
  data3 = [[data['title']],[''],[sid+' '+data['name']],[''],[datetime.date.today().strftime('%Y年%m月%d日')]]
  table3 = Table(data3, colWidths=170*mm)
  table3.setStyle(TableStyle([
            ('FONT', (0, 0), (0, 1), 'HeiseiKakuGo-W5', 24),
            ('FONT', (0, 1), (0, -1), 'HeiseiKakuGo-W5', 16),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ]))
  table3.wrapOn(pdf_canvas, 20*mm, 200*mm)
  table3.drawOn(pdf_canvas, 20*mm, 200*mm)
  pdf_canvas.showPage()

def printResult(pdf_canvas, data):
  for task in data['content']:
    pdf_canvas.setFont("HeiseiKakuGo-W5", 14)
    pdf_canvas.drawString(20*mm, 270*mm, task["q"])
    pdf_canvas.setFont("HeiseiKakuGo-W5", 12)
    if task['t'] == 'ekarte':
      pdf_canvas.drawString(30*mm, 250*mm, 'ID:'+task['a'].患者ID)
      pdf_canvas.drawString(30*mm, 240*mm, '氏名:'+task['a'].患者氏名)
      pdf_canvas.drawString(30*mm, 230*mm, '年齢:'+str(task['a'].年齢())+'歳')
      pdf_canvas.drawString(30*mm, 220*mm, '診療データ数:'+str(len(task['a'].診療データリスト)))
      pos = 210
      if (0 < len(task['a'].診療データリスト)):
        for i in range(0, len(task['a'].診療データリスト)):
          pdf_canvas.drawString(30*mm, pos*mm, '> No.'+str(i+1))
          pdf_canvas.drawString(30*mm, (pos-10)*mm, task['a'].診療データリスト[i].データの取得())
          pos = pos -20
      pdf_canvas.showPage()

def makeReport(data):
  pdf_canvas = canvas.Canvas("./"+data['filename']+".pdf")
  init(pdf_canvas)
  pdf_canvas.setAuthor(data['name'])
  pdf_canvas.setTitle(data['title'])
  pdf_canvas.setSubject("lecture_"+data['lid'])
  printHead(pdf_canvas, data)
  printResult(pdf_canvas, data)
  pdf_canvas.setFont("HeiseiKakuGo-W5", 12)
  pdf_canvas.drawString(20*mm, 270*mm, "感想：")
  pdf_canvas.drawString(20*mm, 250*mm, data['review'])
  pdf_canvas.save()