import requests
from pathlib import Path
from xml.etree.ElementTree import fromstring, ElementTree

import argparse 
import urllib.request
import os


parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('eval_path')
args = parser.parse_args()


data_path = Path(args.data_path)
eval_path = Path(args.eval_path)
eval_path.mkdir(parents=True, exist_ok=True)

for path_wav in list(data_path.glob("*.wav")):
    path_txt = data_path / path_wav.name.replace(".wav", ".txt")
    out_path = eval_path / path_wav.name.replace(".wav", ".TextGrid")
    
    os.system('python3 -m aeneas.tools.execute_task  %s %s "PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX=30|PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX=30|task_language=eng|os_task_file_format=textgrid|is_text_type=plain" %s' % (str(path_wav), str(path_txt), str(out_path)))