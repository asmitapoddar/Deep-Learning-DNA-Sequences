import numpy as np
import pandas as pd
import yaml
import os
import seaborn
import matplotlib.pyplot as plt
import argparse

#/Users/asmitapoddar/Documents/Oxford/Thesis/Genomics Project/Code/attention/lengths/Len60_balanced_AttLSTM[4,128,2,2]_BS32_Adam_29-07_15:59
model_name_save_dir = 'Len40_balanced_AttLSTM[4,64,2,2]_BS32_Adam_02-08_11:31'
with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)

BASE_ATT_PATH = sys_params['ATT_BASE_FOLDER'] + '/end/'

def heatmap(model_name_save_dir, s, e, t):

    att_path = BASE_ATT_PATH + model_name_save_dir
    train_attention_path = att_path + '/' + t
    df_final = pd.DataFrame(columns=['epoch', 'bins', 'values'])
    seaborn.set(font_scale=0.55)

    for epoch in range(s,e+1):
        print('Running epoch {}...'.format(epoch))
        epoch_attention_map = np.loadtxt(train_attention_path+'/attention_map_epoch{}'.format(epoch),
                                         delimiter=',')
        epoch_mean_attention = np.mean(epoch_attention_map, axis=0)
        #print(epoch_mean_attention.shape, epoch_mean_attention)
        len_attention = epoch_mean_attention.shape[0]
        df = {'epoch': ['{0:0=2d}'.format(epoch)]*len_attention,
              'bins': list(range(1,len_attention+1)), 'values': epoch_mean_attention}
        df_final = df_final.append(pd.DataFrame(df), ignore_index=True)

    heatmap_data = pd.pivot_table(df_final, values='values', index=['epoch'], columns='bins')
    seaborn.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=True, yticklabels=True, annot=False)
    #hm.ax_heatmap.set_xticklabels(hm.ax_heatmap.get_xmajorticklabels(), fontsize = 16)
    #sb.palplot(sb.diverging_palette(200, 100, n=11))
    print('Saving heatmap at '+att_path+'/'+t+'_attention_heatmap.png')
    plt.savefig(att_path+'/'+t+'_attention_heatmap.png')
    plt.show()

def plot_all_heatmaps():
    best_epochs = [ 50, 70, 68, 70, 44, 50, 35, 82, 99, 74]
    dirs = os.listdir(BASE_ATT_PATH)

    x = list(range(1,101))
    df = pd.DataFrame({"seq_len": x})

    #Note: Make sure the directories are listed alphabetically in the order of best_epochs array
    for directory in dirs:
        if not os.path.isdir(BASE_ATT_PATH + directory):
            dirs.remove(directory)
    assert len(dirs)==len(best_epochs), "No. of directory mismatch - should be "+str(len(best_epochs))+"; got "+str(len(dirs))

    fig, ax = plt.subplots(len(dirs), sharex=True)
    for i, directory in enumerate(sorted(dirs)):

        print('Running epoch {} for {}...'.format(best_epochs[i], directory))
        epoch_attention_map = np.loadtxt(BASE_ATT_PATH + directory + '/train/positive/attention_map_epoch{}'.format(best_epochs[i])
                                         , delimiter=',')
        epoch_mean_attention = np.mean(epoch_attention_map, axis=0)
        maxValue = np.max(epoch_mean_attention)
        minValue = np.min(epoch_mean_attention)
        OldRange = maxValue - minValue

        epoch_mean_attention = list(map(lambda OldValue: ((OldValue - minValue)/OldRange), epoch_mean_attention))
        len_attention = len(epoch_mean_attention)
        #pad = [None]*(100-len_attention)  # for when boundary is 10 nt before end of sequence
        #pad.extend(epoch_mean_attention)
        pad = [None] * int(50-(len_attention)/2)
        pad.extend(epoch_mean_attention)
        pad.extend([None] * int(50-(len_attention)/2))

        ax[i].plot(x, pad, label=str(len_attention), color='blue')
        ax[i].axvline(x=50, color='red')  # boundary point line
        ax[i].xaxis.set_ticks(np.arange(0, 100, 10))
        ax[i].set(ylabel=len_attention)
        #df[i] = pad

    plt.xlabel("DNA Sequence Length")
    fig.text(0.008, 0.5, 'Normalised Attention Values', va='center', rotation='vertical')
    fig.suptitle('Attention Visualization over DNA Sequence lengths')
    #df.plot(x="seq_len")
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    print('Saving figure at {}..'.format(BASE_ATT_PATH + 'attention_line_plot_all.png'))
    plt.savefig(BASE_ATT_PATH + 'attention_line_plot_all.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention visulaiztion')
    parser.add_argument('-f', '--files', type=str, help='File to plot visualization for (all/file_name)')
    parser.add_argument('-t', '--type', type=str, help='train/val')
    parser.add_argument('-s', '--start', type=int, help='Start epoch')
    parser.add_argument('-e', '--end', type=int, help='End epoch')
    args = parser.parse_args()

    if args.files=='all':
        plot_all_heatmaps()
    else:
        heatmap(args.files, args.start, args.end, args.type)







