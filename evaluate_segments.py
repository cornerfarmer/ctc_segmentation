
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("eval_path")
args = parser.parse_args()

data_path = Path(args.data_path)
eval_path = Path(args.eval_path)

all_diffs = []
for path_wav in data_path.glob("*.wav"):
    with open(str(eval_path / path_wav.name.replace(".wav", ".txt")), "r") as f:
        text = [t.split() for t in f.readlines()[1:]]            
        segments = [(float(t[0]), float(t[1]), " ".join(t[4:])) for t in text]
    
    with open(str(data_path / path_wav.name.replace(".wav", "_ground_truth.txt")), "r") as f:
        text = [t.split() for t in f.readlines()]            
        ground_truth = [(float(t[0]), float(t[1]), " ".join(t[2:])) for t in text]
    
    diffs = []
    for i in range(len(segments)):
        if segments[i][2] != ground_truth[i][2]:
            raise Exception("Text not matching: " + segments[i][2] + " <=> " + ground_truth[i][2])
        
        diffs.append(abs(segments[i][0] - ground_truth[i][0]))
        diffs.append(abs(segments[i][1] - ground_truth[i][1]))

    diffs = np.array(diffs)
    all_diffs.append(diffs)

    print(path_wav.name, diffs.mean(), np.median(diffs), diffs.std())
    
all_diffs = np.concatenate(all_diffs, 0)
print(all_diffs.mean(), np.median(all_diffs), all_diffs.std())

print("<1s", (all_diffs < 1).mean())
print("<0.5s", (all_diffs < 0.5).mean())
plt.hist(all_diffs, density=False, bins=50, range=[0,5])
plt.show()    
