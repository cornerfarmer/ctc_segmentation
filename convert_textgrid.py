import textgrid
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('eval_path')
parser.add_argument('--whole_utt', action='store_true', default=False)
cmdargs = parser.parse_args()


data_path = Path(cmdargs.data_path)
eval_path = Path(cmdargs.eval_path)

for path_wav in data_path.glob("*.wav"):
    if path_wav.name == "MichaelSpecter_2010.wav":
        continue
    path_textgrid = eval_path / (path_wav.name.replace(".wav", ".TextGrid"))
    path_utts = data_path / (path_wav.name.replace(".wav", ".txt"))

    tg = textgrid.TextGrid()
    tg.read(path_textgrid)

    with open(path_utts, "r") as f:
        utts = f.readlines()


    with open(eval_path / (path_wav.name.replace(".wav", ".txt")), "w") as f:
        f.write("\n")
        i = 0
        for utt in utts:

            while tg.tiers[0][i].mark == "":
                i += 1
                
            start_time = tg.tiers[0][i].minTime

            if not cmdargs.whole_utt:
                for word in utt.split():
                    while tg.tiers[0][i].mark == "":
                        i += 1
                    
                    #print(word, tg.tiers[0][i].mark)
                    if "'" in word and not word.replace("'", "") == tg.tiers[0][i].mark and not word == tg.tiers[0][i].mark:
                        continue
                    elif not word == tg.tiers[0][i].mark:
                        word = word.replace("'", "")

                    if word != tg.tiers[0][i].mark:
                        raise Exception("No match: " + word + " <=> " + tg.tiers[0][i].mark)
                    
                    i += 1     
                
                i -= 1 
            else:
                if utt.strip() != tg.tiers[0][i].mark.strip() :
                    raise Exception("No match: " + utt + " <=> " + tg.tiers[0][i].mark)
                
            end_time = tg.tiers[0][i].maxTime
            f.write(str(start_time) + " " + str(end_time) + " 0 | " + utt)
           
            i += 1    
            
