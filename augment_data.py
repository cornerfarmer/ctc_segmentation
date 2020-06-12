import argparse
from pathlib import Path
import wave
import random

parser = argparse.ArgumentParser()
parser.add_argument("src_dir")
parser.add_argument("dest_dir")
args = parser.parse_args()


src = Path(args.src_dir)
dest = Path(args.dest_dir)
dest.mkdir(exist_ok=True)


for path_wav in src.glob("*.wav"):
    w = wave.open(str(path_wav),'r')

    frames = w.readframes(w.getnframes())

    prepend_sec = random.randint(10, 30)
    append_sec = random.randint(10, 30)

    output = wave.open(str(dest / path_wav.name), 'wb')
    output.setparams(w.getparams())
    output.writeframes(frames[-w.getframerate() * prepend_sec*w.getsampwidth():])
    output.writeframes(frames)
    output.writeframes(frames[:w.getframerate() * append_sec*w.getsampwidth()])
    output.close()

    with open(src / path_wav.name.replace(".wav", ".txt"), "r") as f:
        with open(dest / path_wav.name.replace(".wav", ".txt"), "w") as o:
            o.writelines(f.readlines())

    
    with open(src / path_wav.name.replace(".wav", "_ground_truth.txt"), "r") as f:
        with open(dest / path_wav.name.replace(".wav", "_ground_truth.txt"), "w") as o:
            for line in f.readlines():
                parts = line.split()
                o.write(str(float(parts[0]) + prepend_sec) + " " + str(float(parts[1]) + prepend_sec) + " " + " ".join(parts[2:]) + "\n")