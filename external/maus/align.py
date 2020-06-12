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

for path_wav in list(data_path.glob("*.wav"))[1:]:
    path_txt = data_path / path_wav.name.replace(".wav", ".txt")
    out_path = eval_path / path_wav.name.replace(".wav", ".TextGrid")
    print(path_wav.name)

    result = os.popen("curl -X POST -H 'content-type: multipart/form-data' -F PIPE=G2P_CHUNKER_MAUS -F SIGNAL=@%s -F LANGUAGE=deu-DE -F TEXT=@%s 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runPipeline'" % (str(path_wav), str(path_txt))).read()

    print(result)
    tree = ElementTree(fromstring(result))

    root = tree.getroot()
    downloadlink = root.find("downloadLink").text
    urllib.request.urlretrieve(downloadlink, str(out_path))