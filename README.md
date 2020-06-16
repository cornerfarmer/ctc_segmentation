# CTC segmentation

```
source ../../tools/venv/bin/activate
```

## Setup

Install ESPnet (tested v0.7.0)

https://espnet.github.io/espnet/installation.html

Clone this repo into egs

## Setup data

```
wget https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release1.tar.gz
tar xvf TEDLIUM_release1.tar.gz
```

```
python convert_ted.py TEDLIUM_release1/dev data/tedlium_dev
python convert_ted.py TEDLIUM_release1/test data/tedlium_test
```

## Download model

tedlium2.rnn.v2
https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf

```
tar xvf model.streaming.v2.tar.gz --one-top-level
mkdir -p exp/tedlium2_rnn
mv model.streaming.v2/exp/train_trim_sp_pytorch_train2/results/* exp/tedlium2_rnn/
mv model.streaming.v2/data/train_trim_sp/cmvn.ark exp/tedlium2_rnn/
```

## Decode audio

if transformer => --split
if use gpu => --gpu

```
python decode.py exp/tedlium2_rnn/model.acc.best exp/tedlium2_rnn/cmvn.ark data/tedlium_dev eval/tedlium_dev_tedlium2_rnn
```

## Segment audio

```
python align.py exp/tedlium2_rnn/model.acc.best data/tedlium_dev eval/tedlium_dev_tedlium2_rnn
```