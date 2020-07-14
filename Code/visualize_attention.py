import numpy as np
import pandas as pd
import yaml
import seaborn
import matplotlib.pyplot as plt

model_name_save_dir = 'attention_Attention[4,16,3,2]_BS32_Adam_14-07_22:03'

with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)
att_path = sys_params['ATT_BASE_FOLDER'] + '/' + model_name_save_dir
train_attention_path = att_path + '/train'
df_final = pd.DataFrame(columns = ['epoch', 'bins', 'values'])
for epoch in range(0,35):
    epoch_attention_map = np.loadtxt(train_attention_path+'/attention_map_epoch{}'.format(epoch),
                                     delimiter=',')
    epoch_mean_attention = np.mean(epoch_attention_map, axis=0)
    len_attention = epoch_mean_attention.shape[0]
    df = {'epoch': ['epoch{}'.format(epoch)]*len_attention,
          'bins': list(range(1,101)), 'values': epoch_mean_attention}
    df_final = df_final.append(pd.DataFrame(df), ignore_index=True)

heatmap_data = pd.pivot_table(df_final, values='values', index=['epoch'], columns='bins')
#sb.set(font_scale=2)
seaborn.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=True, yticklabels=True, annot=False)
#sb.palplot(sb.diverging_palette(200, 100, n=11))
plt.savefig(train_attention_path+'attention_heatmap.png')
plt.show()





