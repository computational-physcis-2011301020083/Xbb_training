:<<EOF
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



