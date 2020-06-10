import argparse
from pathlib import Path
import shutil
from sphfile import SPHFile


parser = argparse.ArgumentParser()
parser.add_argument("src_dir")
parser.add_argument("dest_dir")
args = parser.parse_args()

src = Path(args.src_dir)
dest = Path(args.dest_dir)


for file_sph in (src / "sph").iterdir():
    file_stm = src / "stm" / file_sph.name.replace(".sph", ".stm")

    dest.mkdir(parents=True, exist_ok=True)

    sph = SPHFile(str(file_sph))
    sph.write_wav(str(dest / file_sph.name.replace(".sph", ".wav")))

    with open(file_stm, "r") as f:
        output = []
        ground_truth = []
        for line in f.readlines():
            line_parts = line.split()
            line = " ".join(line_parts[6:])
            if line != "ignore_time_segment_in_scoring":
                output.append(line)
                ground_truth.append(" ".join([line_parts[3], line_parts[4]] + line_parts[6:]))

    with open(str(dest / file_sph.name.replace(".sph", ".txt")), "w") as f:
        f.write("\n".join(output))

    with open(str(dest / file_sph.name.replace(".sph", "_ground_truth.txt")), "w") as f:
        f.write("\n".join(ground_truth))




