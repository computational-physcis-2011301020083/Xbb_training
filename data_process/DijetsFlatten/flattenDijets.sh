j=1
for i in $(ls ../DataVRGhost/ReducedDijets/*h5)
do
	path=$i
	echo $j,$path
	python flattenDijets.py --path $path
	let j=j+1
done


path="../DataVRGhost/FlattenData3a/MergedDijets/"
for i in 361023 361024 361025 361026 361027 361028 361029 361030
do
	python MergeDijetsNp.py --path $path --dsid $i
done

for i in $(ls ../DataVRGhost/FlattenData3a/MergedDijetsDSID/)
do
	path="../DataVRGhost/FlattenData3a/MergedDijetsDSID/"$i
	echo $path
	python subDijet.py --path $path
done

path="../DataVRGhost/FlattenData3a/ReducedDijetsDSID/"
python MergeDijetsNp1.py --path $path


