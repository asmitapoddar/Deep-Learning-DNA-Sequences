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
#for f in */dna_seq; do (cat "${f}"; echo) >> dna_seq; done
#for f in */y_label; do (cat "${f}"; echo) >> y_label; done
#wc -l dna_seq  # 233629
#wc -l y_label

sudo python3 meta_info_stitch.py --write_path '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/60'
sudo python3 encode.py --write_path '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/60'

sudo python3 subset.py -i '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/70/' -n 50000
sudo python3 encode.py --write_path '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/70' --in_path 'sub'
