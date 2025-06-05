import os
import re
import datetime
import subprocess
from google.colab import _message

###############　課題・練習問題の回答収集　##################
def getTaskResults(note_content):
  results = []
  # 管理用
  is_task = False
  is_prac = False
  idlist = {}
  #
  for cell in note_content['ipynb']['cells']:
    if 'task-end' in cell['source'][0]:
      is_task = False
    elif 'prac-end' in cell['source'][0]:
      is_prac = False
    elif 'task-start' in cell['source'][0]:
      matched = re.search('.*id="(.+)".*', cell['source'][0])
      last_id = matched.groups()[0]
      idlist[last_id] = 1
      is_task = True
      print(last_id, '読み取り中')
    elif 'prac-start' in cell['source'][0]:
      matched = re.search('.*id="(.+)".*', cell['source'][0])
      last_id = matched.groups()[0]
      idlist[last_id] = 1
      is_prac = True
      print(last_id, '読み取り中')
    # 採点対象のコードセル
    if (is_task or is_prac) and cell['cell_type'] == 'code':
      res = {
          'id': f"{last_id}{'-'+str(idlist[last_id]) if idlist[last_id] > 1 else ''}",
          'src': '<br/>'.join(cell['source']).replace('\n', ''),
          'type': 'task' if is_task else 'practice'
      }
      idlist[last_id] += 1
      # 出力がなければ未実行
      if len(cell['outputs']) == 0:
        res['exec'] = False
      else:
        res['exec'] = True
        # 最終実行結果の取得
        last_exe = cell['outputs'][-1]
        # エラーの場合
        if last_exe['output_type'] == 'error':
          res['return'] = 'error'
          res['stderr'] = last_exe['evalue']
        # HTML形式での出力
        elif last_exe['output_type'] == 'display_data':
          res['return'] = 'ok'
          if 'text/html' in last_exe['data']:
            res['output'] = ''.join(last_exe['data']['text/html']).replace('\n', '')
        # 標準出力
        else:
          res['return'] = 'ok'
          res['output'] = '<br/>'.join(last_exe['text']).replace('\n', '')
      # 
      results.append(res)
  return results
##########################################################

####################　レポートの作成　######################
def makeReport(dir, filename, content, info):
  # 作成日時
  now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST'))
  # 回答結果
  print('読み込み開始')
  results = getTaskResults(content)
  print('読み込み完了')
  # 一次ファイルの作成
  with open(f'{dir}/report.html', 'w') as f:
    f.write(f'''
    <html><body>
    <h2 style="text-align:center;">プログラミング I (Python実践演習)レポート</h2>
    <div style="text-align:right; font-size:18px;"><p>学籍番号：{info['id']}</p><p>氏名：{info['name']}</p><p>作成日：{now.strftime("%Y/%m/%d")}</p></div>
    ''')
    for res in results:
      if res['exec']:
        f.write(f'''
        <div style="border: 0.5px solid black; border-radius:6px; padding: 8px; margin: 5px;">
        <h3>{res['id']}</h3>
        { '<h4>コード：</h4><div style="background-color: lightgray; padding: 10px;">'+res['src']+'</div>' if res['type'] == 'practice' else ''}
        <h4>結果：</h4>
        { '<span style="font-size:14px; color:red">エラー</span><br/>' + res['stderr'] if res['return'] == 'error' else res['output'] }
        </div>
        ''')
      else:
        f.write(f'''
        <div >{res['id']}</div>
        <div style="font-color:red">未実行</div>
        ''')

    f.write(f'''</body></html>''')
  cmd = f"google-chrome --disable-gpu --headless --no-margins --no-pdf-header-footer --no-sandbox --print-to-pdf='/content/{filename}' {dir}/report.html"
  res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  if res.returncode == 0:
    return True
  else:
    print(res.stderr)
    return False
##########################################################
