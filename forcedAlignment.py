import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},build_dir="build", build_in_temp=False)

from alignSearch import cython_fill_table

import sys

import torch

sys.path.append("../../../espnet")
from espnet.asr.asr_utils import get_model_conf

import os
from pathlib import Path
from time import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("data_path")
parser.add_argument("eval_path")
parser.add_argument('--id', type=int, default=None)
parser.add_argument('--start_win', type=int, default=8000)
parser.add_argument('--skip_prob', type=float, default=-3)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

max_prob = -10000000000.0

def recognize(lpz, char_list, ground_truth, utt_begin_indices, skip_prob):   
    blank = 0
    print(lpz.shape[1], len(ground_truth))
    if len(ground_truth) > lpz.shape[0] and skip_prob <= max_prob:
        raise AssertionError("Audio is shorter than text!")
    window_len = args.start_win

    while True:
        table = np.zeros([min(window_len, lpz.shape[0]), len(ground_truth)], dtype=np.float32)
        table.fill(max_prob)
        start = time()
        offsets = np.zeros([len(ground_truth)], dtype=np.int)
        t, c = cython_fill_table(table, lpz.astype(np.float32), np.array(ground_truth), offsets, np.array(utt_begin_indices), blank, skip_prob)
        print(time() - start)

        print("Max prob: " + str(table[:, c].max()) + " at " + str(t))
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        char_skips = np.zeros([table.shape[1]])
        delta = table.shape[0] / float(table.shape[1])
        char_list = [''] * lpz.shape[0]
        current_prob_sum = 0
        try:
            while t != 0 or c != 0:
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = max_prob
                for s in range(ground_truth.shape[1]): 
                    if ground_truth[c, s] != -1:                   
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = lpz[t + offsets[c], ground_truth[c, s]] if c > 0 else max_prob
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s

                        max_lpz_prob = max(max_lpz_prob, switch_prob)

                
                stay_prob = max(lpz[t + offsets[c], blank], max_lpz_prob) if t > 0 else max_prob
                est_stay_prob = table[t, c] - table[t - 1, c]
                
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    if c > 0:
                        for s in range(0, min_s + 1):
                            timings[c - s] = (offsets[c] + t) * 10 * 4 / 1000
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_list[offsets[c] + t] = train_args.char_list[ground_truth[c, min_s]]
                        current_prob_sum = 0
                        #table[table[:, c].argmax(), c] *= -1
                        #table[t, c] = np.abs(table[t, c])

                    c -= 1 + min_s
                    t -= 1 - offset
                 
                else:
                    char_probs[offsets[c] + t] = stay_prob
                    char_list[offsets[c] + t] = "ε"
                    t -= 1
        except IndexError:
            window_len *= 2
            print("IndexError: Trying with win len: " + str(window_len))
            if window_len < 100000:
                continue
            else:
                raise

        break

    return timings, char_probs, char_skips, char_list


model_path = args.model_path
model_conf = None

# read training config
idim, odim, train_args = get_model_conf(model_path, model_conf)
#train_args.char_list[21] = " "
space_id = train_args.char_list.index('▁')
train_args.char_list[0] = "ε"
train_args.char_list = [c.lower() for c in train_args.char_list]
print(train_args.char_list)

max_char_len = max([len(c) for c in train_args.char_list])

data_path = Path(args.data_path)
eval_path = Path(args.eval_path)

for path_wav in data_path.glob("*.wav"):     

    chapter_sents = data_path / path_wav.name.replace(".wav", ".txt")
    chapter_prob = eval_path / path_wav.name.replace(".wav", ".npz")
    out_path = eval_path / path_wav.name.replace(".wav", ".txt")

    with open(str(chapter_sents), "r") as f:
        text = [t.strip() for t in f.readlines()]

    lpz = np.load(str(chapter_prob))["arr_0"]
    #chars = np.array(train_args.char_list)
    #for i in range(500):
    #    print(chars[(-lpz[i]).argsort()[:3]], lpz[i][(-lpz[i]).argsort()[:3]])
    #exit(0)

    print("Syncing " + str(path_wav))

    ground_truth = "#"#[1000000000]
    utt_begin_indices = []
    for utt in text:
        if ground_truth[-1] != " ":
            ground_truth += " "

        utt_begin_indices.append(len(ground_truth) - 1)

        #utt, _ = clean_text(utt, False, source.name == "librivox")
        #print(utt)
        for char in utt:
            if char.isspace():
                if ground_truth[-1] != " ":
                    ground_truth += " "
            elif char in train_args.char_list and char not in [ ".", ",", "-", "?", "!", ":", "»", "«", ";", "'", "›", "‹", "(", ")"]:
                ground_truth += char

    if ground_truth[-1] != " ":
        ground_truth += " "
    utt_begin_indices.append(len(ground_truth) - 1)
        
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i-s < 0:
                continue
            span = ground_truth[i-s:i+1]
            span = span.replace(" ", '▁')
            if span in train_args.char_list:
                ground_truth_mat[i, s] = train_args.char_list.index(span)                                
        
    try:
        timings, char_probs, char_skips, char_list = recognize(lpz, train_args.char_list, ground_truth_mat, utt_begin_indices, max_prob)
    except AssertionError:
        print("Skipping: Audio is shorter than text")
        continue

    for i in range(1000):#len(char_probs)):
        print(i * 10 * 4 / 1000, char_probs[i], char_list[i])

    #print(timings[-20:])
    #print(ground_truth[-20:])

    with open(str(out_path), 'w') as outfile:
        outfile.write(str(path_wav.name) + '\n')

        def compute_time(index, type="center"):
            middle = (timings[index] + timings[index - 1]) / 2
            if type == "begin":
                return max(timings[index + 1] - 0.5, middle)
            elif type == "end":
                return min(timings[index - 1] + 0.5, middle)

        for i in range(len(text)):
            if char_skips[utt_begin_indices[i + 1]]:
                continue
            start = compute_time(utt_begin_indices[i], "begin")
            end = compute_time(utt_begin_indices[i + 1], "end")
            start_t = int(round(start * 1000 / 40))
            end_t = int(round(end * 1000 / 40))

            n = 30
            if end_t == start_t:
                min_avg = 0
            elif end_t - start_t <= n:
                min_avg = char_probs[start_t:end_t].mean()
            else:
                min_avg = 0
                for t in range(start_t, end_t - n):
                    min_avg = min(min_avg, char_probs[t:t + n].mean())
                    
            outfile.write(str(start) + " " + str(end) + " " + str(min_avg) + " | " + text[i] + '\n')

        if args.debug:                            
            char_max_list = np.array(train_args.char_list)[lpz.argmax(axis=-1)]
            char_max_prob = lpz.max(axis=-1)

            with open("fa-debug", 'w') as outfile:
                for t in range(len(char_list)): 
                    outfile.write(char_list[t] + "\t" + '%.2f' % char_probs[t] + "\t" + char_max_list[t] + "\t" + '%.2f' % char_max_prob[t] + '\n')

    #np.stack([char_probs[start_t:end_t], np.array(train_args.char_list)[lpz[start_t:end_t].argmax(axis=-1)], np.array(train_args.char_list)[np.where(lpz[start_t:end_t] == np.expand_dims(char_probs[start_t:end_t], -1))[1]]])

    # with open(str(path_segments / ('chapter-' + str(chapter_num) + ".debug.txt")), 'w') as outfile:
    #    for i in range(len(text)):
    #        outfile.write(str(compute_time(utt_begin_indices[i], "right")) + "\t" + str(compute_time(utt_begin_indices[i + 1] if i < len(text) - 1 else -1, "left")) + "\t" + str(i) + "\n")
