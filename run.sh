#How to run

:<<EOF
path="Dijets sample path"
python CalculatedDijetsWeights.py --path $path
#EOF

path="Dijets sample path"
python labelDijetsDatasets.py --path $path
path="Top sample path"
python labelTopDatasets.py --path $path
path="Hbb sample path"
python labelHbbDatasets.py --path $path
#EOF

for i in TopSamplePath HbbSamplePtah
do 
  python MergeDatasets.py --path $path
done

path="Dijets sample path"
python MergeDijetsDatasets.py --path $path
#EOF

for i in $(ls MergedData)
do
  path="MergedData/"$i
  echo $path
  python flatten.py --path $path
done
#EOF

for i in $(ls FlattenData)
do 
  path="FlattenData/"$i
  echo $path
  python split.py --path $path
done
#EOF

path="SplitData"
python calculate.py --path $path
EOF

path="SplitData"
python train.py --path $path

