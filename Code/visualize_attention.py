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

BASE_ATT_PATH = sys_params['ATT_BASE_FOLDER'] + '/final/'

def heatmap(model_name_save_dir, s, e, t):

    att_path = BASE_ATT_PATH + model_name_save_dir
    train_attention_path = att_path + '/' + t
    df_final = pd.DataFrame(columns=['epoch', 'bins', 'values'])

    for epoch in range(s,e+1):
        print('Running epoch {}...'.format(epoch))
        epoch_attention_map = np.loadtxt(train_attention_path+'/attention_map_epoch{}'.format(epoch),
                                         delimiter=',')
        epoch_mean_attention = np.mean(epoch_attention_map, axis=0)
        #print(epoch_mean_attention.shape, epoch_mean_attention)
        len_attention = epoch_mean_attention.shape[0]
        df = {'epoch': ['epoch{}'.format(epoch)]*len_attention,
              'bins': list(range(1,len_attention+1)), 'values': epoch_mean_attention}
        df_final = df_final.append(pd.DataFrame(df), ignore_index=True)

    heatmap_data = pd.pivot_table(df_final, values='values', index=['epoch'], columns='bins')
    #sb.set(font_scale=2)
    seaborn.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=True, yticklabels=True, annot=False)
    #sb.palplot(sb.diverging_palette(200, 100, n=11))
    print('Saving heatmap at '+att_path+'/'+t+'_attention_heatmap.png')
    plt.savefig(att_path+'/'+t+'_attention_heatmap.png')
    plt.show()

def plot_all_heatmaps():
    best_epochs = [ 45, 83, 19, 35, 70, 60, 85, 44, 28]
    dirs = os.listdir(BASE_ATT_PATH)
    #dir_paths = list(map(lambda name: os.path.join(att_path, name), dirs))

    x = list(range(1,101))
    df = pd.DataFrame({"seq_len": x})
    for directory in dirs:
        if not os.path.isdir(BASE_ATT_PATH + directory):
            dirs.remove(directory)

    assert len(dirs)==len(best_epochs), "No. of directory mismatch - should be "+str(len(best_epochs))+"; got "+str(len(dirs))
    fig, ax = plt.subplots(len(dirs))
    print(sorted(dirs))
    for i, directory in enumerate(sorted(dirs)):

        print('Running epoch {} for {}...'.format(best_epochs[i], directory))
        epoch_attention_map = np.loadtxt(BASE_ATT_PATH + directory + '/train/attention_map_epoch{}'.format(best_epochs[i])
                                         , delimiter=',')
        epoch_mean_attention = np.mean(epoch_attention_map, axis=0)
        maxValue = np.max(epoch_mean_attention)
        minValue = np.min(epoch_mean_attention)
        OldRange = maxValue - minValue

        epoch_mean_attention = list(map(lambda OldValue: ((OldValue - minValue)/OldRange), epoch_mean_attention))

        len_attention = len(epoch_mean_attention)
        pad = [None]*(100-len_attention)  # todo for when boundary is in the middle for different lengths
        pad.extend(epoch_mean_attention)

        ax[i].plot(x, pad, label='Prices 2008-2018', color='blue')
        ax[i].axvline(x=90, color='red')
        ax[i].xaxis.set_ticks(np.arange(0, 100, 10))
        #ax.plot(years_a30, Geb_a30, label='Prices 2010-2018', color='red')
        df[i] = pad

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("A test graph")
    #df.plot(x="seq_len")
    plt.legend()
    plt.savefig(BASE_ATT_PATH + 'attention_line_plot_all.png')
    plt.show()


def line_plot():
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention visulaiztion')
    parser.add_argument('-f', '--files', type=str, help='File to plot visualization for (all/file_name)')
    parser.add_argument('-l', '--line_plot', type=bool, help='Whether to make line plot')
    parser.add_argument('-t', '--type', type=str, help='train/val')
    parser.add_argument('-s', '--start', type=int, help='Start epoch')
    parser.add_argument('-e', '--end', type=int, help='End epoch')
    args = parser.parse_args()

    if args.files=='all':
        plot_all_heatmaps()
    else:
        heatmap(args.files, args.start, args.end, args.type)







