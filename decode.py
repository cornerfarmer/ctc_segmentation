import numpy as np

import sys

import torch

sys.path.append("../../../espnet")
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.pytorch_backend.asr import load_trained_model
import logging
#from espnet.transform.transformation import using_transform_config
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.nets.pytorch_backend.e2e_asr import E2E
from espnet.asr.asr_utils import torch_load

from heapq import heappop, heappush
import os
from pathlib import Path
from time import time
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('cmvn_path')
parser.add_argument('data_path')
parser.add_argument('output_path')
parser.add_argument('--split', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)
cmdargs = parser.parse_args()



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def recognize(model, x):
    prev = model.training
    model.eval()
    enc_output = model.encode(x).unsqueeze(0)
    print(enc_output.shape)
    lpz = model.ctc.log_softmax(enc_output)[0].cpu().numpy()
    print(lpz.shape)
    return lpz


# debug mode setting
# 0 would be fastest, but 1 seems to be reasonable
# considering reproducibility
# remove type check
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # https://github.com/pytorch/pytorch/issues/6351

logging.info('torch type check is disabled')
# use deterministic computation or not

model_path = cmdargs.model_path
model_conf = None
# read training config

# load trained model parameters
logging.info('reading model parameters from ' + model_path)
model, train_args = load_trained_model(model_path)
model.eval()
args = AttrDict()
args.update({
    "ctc_weight": 0.5,
    "beam_size": 1,
    "penalty": 0.0,
    "maxlenratio": 0.0,
    "minlenratio": 0.0,
    "nbest": 1
})
model.recog_args = args

if cmdargs.gpu:
    gpu_id = range(1)
    logging.info('gpu id: ' + str(gpu_id))
    model.cuda()
else:
    torch.set_num_threads(4)

data_path = Path(cmdargs.data_path)
output_path = Path(cmdargs.output_path)
output_path.mkdir(parents=True, exist_ok=True)

Path("tmp").mkdir(parents=True, exist_ok=True)

for path_wav in data_path.glob("*.wav"):
    output_file = output_path / (path_wav.name.replace(".wav", ".npz"))
    print("Predicting: " + path_wav.name)

    with open("tmp/wav.scp", "w+") as f:
        f.write("file " + str(path_wav.resolve()))  # data/wav/german-speechdata-package-v2/test/2015-02-10-11-31-32_Kinect-Beam.wav crawl/tmp/test.wav

    os.system("./fbanks.sh " + cmdargs.cmvn_path)
    print("Finished fbanks")

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False)
        #preprocess_conf=train_args.preprocess_conf)

    with torch.no_grad():
        data = {"input": [{"name": "input1", "feat": str(Path("tmp/feats.1.ark:5").resolve())}]}

        #with using_transform_config({'train': True}):
        full_feat = load_inputs_and_targets([("data", data)])[0][0]

        if cmdargs.split:
                
            all_probs = []
            for i in range(len(full_feat) // 16000 + 1):
                feat = full_feat[i * 16000: (i + 1) * 16000]

                probs = recognize(model, feat)
                all_probs.append(probs)
                #exit(0)
        
            probs = np.concatenate(all_probs, 0)
        else:
            probs = recognize(model, full_feat)

        print("Finished recog")
        print(probs.shape)

        tmp_path = Path("tmp/output.npz")
        np.savez(tmp_path, probs)
        shutil.move(tmp_path, output_file)
        print("Finished saving")
