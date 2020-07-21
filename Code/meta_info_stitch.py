import os
import argparse

total = 0
intron_exon = 0
exon = 0
intron = 0

parser = argparse.ArgumentParser(description='Create meta-info')
parser.add_argument('--write_path', type=str, help='write_path for meta-info file')
args = parser.parse_args()
#directory= '/mnt/sdc/asmita/Code/Data/all/boundaryCertainPoint_orNot_2classification/60/'
directory = args.write_path

for dir in os.listdir(directory):
    if os.path.isfile(directory + '/' + dir):
        print('not dir', dir)
        continue
    with open(directory + '/' + dir + '/info.log', 'r') as f:
        for line in f:
            if "Total no. of samples in training set" in line:
                total += (int)(line[39:])
                print('total', total)
            if "No. samples containing intron-exon boundary" in line:
                intron_exon += (int)(line[46:])
                print('intron-exon', intron_exon)
            if "No. of pure exon samples" in line:
                exon += (int)(line[27:])
                print('exon', exon)
            if "No. of pure intron samples" in line:
                intron += (int)(line[29:])
                print('intron', intron)

with open(directory+'info.log', 'w') as f:
    f.write('Total no. of samples in training set: ' +str(total))
    f.write('\n No. samples containing intron-exon boundary: ' + str(intron_exon))
    f.write('\n No. of pure exon samples: '+ str(exon))
    f.write('\n No. of pure intron samples: '+ str(intron))
f.close()