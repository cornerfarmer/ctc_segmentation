
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.stats as st
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set(style="whitegrid")

import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("data_path")
#parser.add_argument("eval_path")
#args = parser.parse_args()

result = Path("result")
result.mkdir(exist_ok=True)

models = [
            "aeneas_%s",
             "maus_%s",
              "gentle_%s",
               "tedlium_%s_lstm",
                "tedlium_%s_transformer", 
                "tedlium_%s_transformer_tedlium_16",
                 "tedlium_%s_transformer_tedlium_8",
                  "tedlium_%s_transformer_tedlium_4",
                   "tedlium_%s_transformer_tedlium_2",
                   "tedlium_%s_transformer_tedlium_1"
            ]
models = ["tedlium_%s_lstm","tedlium_%s_transformer_tedlium_2","tedlium_%s_transformer_libri_2","gentle_%s","aeneas_%s","maus_%s"]#["tedlium_%s_transformer_tedlium_2","gentle_%s"]
models = ["tedlium_%s_lstm","gentle_%s"]#["tedlium_%s_transformer_tedlium_2","gentle_%s"]

fig = plt.figure(figsize=(8.27,2.5))
gs = gridspec.GridSpec(1, 2, figure=fig)

with open(result / "metrics", "w") as o:
    o.write("      mean    median    std    <1s    <0.5s\n")

    for b, boundary in enumerate(["start", "end"]):

        for data_pack in [("test_augm", "dev_augm")]:#, "test",("test", "dev")

            ax0 = fig.add_subplot(gs[0, b])

            for m, model in enumerate(models):

                all_diffs = []
                for data in data_pack:
                    eval_path = Path("eval") / (model % data)
                    data_path = Path("data") / ("tedlium_%s" % data)

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
                            
                            if boundary == "start":
                                diffs.append(abs(segments[i][0] - ground_truth[i][0]))
                            else:
                                diffs.append(abs(segments[i][1] - ground_truth[i][1]))

                        diffs = np.array(diffs)
                        all_diffs.append(diffs)

                        #print(path_wav.name, diffs.mean(), np.median(diffs), diffs.std())

                all_diffs = np.concatenate(all_diffs, 0)
                o.write(data_pack[0] + " " + model + " " + ("%.3f" % all_diffs.mean()) + " " + ("%.3f" % np.median(all_diffs)) + " " + ("%.3f" % all_diffs.std()) + " " + ("%.3f" % (all_diffs < 1).mean()) + " " + ("%.3f" % (all_diffs < 0.5).mean()) + "\n")
    #
                all_diffs = all_diffs[np.logical_and(0 < all_diffs, all_diffs < 2)]

                sns.distplot(all_diffs, hist=False, bins=100, norm_hist=True,label={"tedlium_%s_lstm": "ours", "gentle_%s": "gentle"}[model], kde_kws={'linestyle':'--' if m != 0 else '-', 'color':'black'})
            
            
            ax0.set_xlim(0,1.25)
            ax0.set_ylim(0,3.1)
            ax0.legend()   
            ax0.set_ylabel('Density')
            ax0.set_xlabel('Deviation to manually labeled segments')
    fig.tight_layout()
    fig.savefig("result/histo.eps") 
            #exit(0)
