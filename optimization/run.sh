#!/bin/bash
cd /global/project/projectdirs/atlas/massDecorrelatedXbb/adversarial-wei1
source activate.sh
cd /global/project/projectdirs/atlas/massDecorrelatedXbb/Xbb_training/optimizationStand

i=1
for layer in 7 9
do
  for nodes in 32 64 96 128
  do
    for dropout in 0.0 0.25 0.5
    do
      for rate in 0.1 0.01 0.001 0.0001
      do
        for decay in 0.01 0.001 0.0001 0.00001
        do
          echo $i
	  let i=i+1
          python optimize.py --layers $layer --nodes $nodes --dropout $dropout --rate $rate --decay $decay
        done
      done
    done
  done
done
