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
#for f in */dna_seq_start; do (cat "${f}"; echo) >> dna_seq_start; done
#for f in */y_label_start; do (cat "${f}"; echo) >> y_label_start; done
#wc -l dna_seq_start  # 233629
#wc -l y_label_start

sudo python3 meta_info_stitch.py --write_path '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/60'
sudo python3 encode.py --write_path '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/60'

