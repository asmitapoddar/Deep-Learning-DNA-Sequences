# sh process.sh

for((i=1;i<=22;i++))
do
  cd raw_data/
  gunzip chr$i.fa.gz
  cd ..
  python3 read_annotations.py --no $i  # create txt and annotation file for chromosome
  python3 create_json_cds.py --no $i  # create json file for chromosome
done

python3 generate_entire_dataset.py
#sudo su
#cat */dna_seq_start >dna_seq_start
#cat */y_label_start >y_label_start
#wc -l dna_seq_start
#wc -l y_label_start

python3 meta_info_stitch.py

