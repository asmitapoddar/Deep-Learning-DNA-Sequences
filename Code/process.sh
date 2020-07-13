# sh process.sh

for((i=1;i<=22;i++))
do
  cd raw_data/
  gunzip chr$i.fa
  cd ..
  python3 read_annotations.py --no $i  # create txt and annotation file for chromosome
  python3 create_json_cds.py --no $i  # create json file for chromosome
done
