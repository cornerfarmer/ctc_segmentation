#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh

fbank_config=conf/fbank.conf
pitch_config=conf/pitch.conf
scp=tmp/wav.scp
paste_length_tolerance=2
compress=true

fbank_feats="ark:compute-fbank-feats --verbose=2 --config=$fbank_config scp,p:tmp/wav.scp ark:- |"
pitch_feats="ark,s,cs:compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config scp,p:tmp/wav.scp ark:- | process-kaldi-pitch-feats ark:- ark:- |"

paste-feats --length-tolerance=$paste_length_tolerance "$fbank_feats" "$pitch_feats" ark:- | \
copy-feats --compress=$compress ark:- \
  ark,scp:tmp/raw_fbank_pitch.ark,tmp/raw_fbank_pitch.scp \
 || exit 1;
# crawl/exp/train_trim_sp_pytorch_train2/cmvn.ark
dump.sh --cmd "$train_cmd" --do_delta false \
          tmp/raw_fbank_pitch.scp $1 tmp/dump.log \
          tmp
