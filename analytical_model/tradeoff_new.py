import csv
from scipy.stats import gmean
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm
import colors

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

series = {}
with open('./data/tradeoff.csv') as f:
  seriest = {}
  cr = csv.reader(f, delimiter='\t')
  lines = [l for l in cr]
  w = len(lines[0])
  for i in range(w):
    seriest[i] = []
  
  for l in lines[1:]:
    for i in range(w):
      seriest[i] += [float(l[i])]

  for i, j in enumerate(lines[0]):
    series[j] = seriest[i]

colorMap = colors.category2


fig, axs = plt.subplots(2, 1)
fig.set_size_inches(10.5, 2.5)
plt.subplots_adjust(top=1.2, wspace=0.12, hspace=0.56)

mixtral_dense_he = series['MIXDENSE-HE']
mixtral_sparse_he = series['MIXSPARSE-HE']

mixtral_dense_gs = series['MIXDENSE-GS']
mixtral_sparse_gs = series['MIXSPARSE-GS']

blackmamba_dense_he = series['BLKDENSE-HE']
blackmamba_sparse_he = series['BLKSPARSE-HE']

blackmamba_dense_gs = series['BLKDENSE-GS']
blackmamba_sparse_gs = series['BLKSPARSE-GS']

x = series['Accuracy']

graph_font_size = 16

axs[0].plot(x, mixtral_dense_he, '-o', color='#377eb8', lw=2.5, label='Mixtral-dense-HE')
axs[0].plot(x, mixtral_sparse_he, '--o', color='#80b1d3', lw=2.5, label='Mixtral-sparse-HE')

axs[0].plot(x, mixtral_dense_gs, '-o', color='#e41a1c', lw=2.5, label='Mixtral-dense-GS')
axs[0].plot(x, mixtral_sparse_gs, '--o', color='#fb8072', lw=2.5, label='Mixtral-sparse-GS')

axs[0].legend(fontsize=graph_font_size-5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
axs[0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=graph_font_size-3)
axs[0].set_xticklabels(range(-2, 11, 2), fontsize=graph_font_size-3)
axs[0].set_ylim(0, 1.2)

font0 = matplotlib.font_manager.FontProperties()
font0.set_weight('bold')
font0.set_size(graph_font_size-2)
# axs[0].xlabel(r"$\mathbf{log_2(}$num of ways$\mathbf{)}$", fontsize=graph_font_size-3, font_properties=font0)
# axs[0].set_xlabel(r"Epoch", fontsize=graph_font_size-3, font_properties=font0)
# axs[0].set_ylabel("Accuracy", fontsize=graph_font_size-3,
          #  font_properties=font0)



axs[1].plot(x, blackmamba_dense_he, '-o', color='#377eb8', lw=2.5, label='Blackmamba -dense-HE')
axs[1].plot(x, blackmamba_sparse_he, '--o', color='#80b1d3', lw=2.5, label='Blackmamba -sparse-HE')

axs[1].plot(x, blackmamba_dense_gs, '-o', color='#e41a1c', lw=2.5, label='Blackmamba -dense-GS')
axs[1].plot(x, blackmamba_sparse_gs, '--o', color='#fb8072', lw=2.5, label='Blackmamba -sparse-GS')

axs[1].legend(fontsize=graph_font_size-5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
axs[1].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=graph_font_size-3)
axs[1].set_xticklabels(range(-2, 11, 2), fontsize=graph_font_size-3)
axs[1].set_ylim(0, 0.6)
axs[1].set_ylim(0, 0.5)

font0 = matplotlib.font_manager.FontProperties()
font0.set_weight('bold')
font0.set_size(graph_font_size-2)
# axs[1].xlabel(r"$\mathbf{log_2(}$num of ways$\mathbf{)}$", fontsize=graph_font_size-3, font_properties=font0)
# axs[1].set_xlabel(r"Epoch", fontsize=graph_font_size-3, font_properties=font0)
# axs[1].set_ylabel("                                   Accuracy", fontsize=graph_font_size-3,
#            font_properties=font0)


fig.text(0.46,0,"Epoch", va='center', fontsize=graph_font_size-2)
fig.text(0.02,0.62,"Accuracy", rotation=90, va='center', fontsize=graph_font_size-2)



# axs[0].text(0.85, 1.15,
#         "Palermo (secure)", fontsize=graph_font_size-5,rotation=00, weight='bold')
# axs[0].arrow(0.2, 1.05,
#           0.55, 0.12,
#           length_includes_head=True,
#           head_width=0, head_length=0, lw=2.25)

# axs[0].text(0.85, 1.605,
#         "RingORAM (secure)", fontsize=graph_font_size-5,rotation=00, weight='bold')
# axs[0].arrow(0.2, 2.15,
#           0.55, -0.45,
#           length_includes_head=True,
#           head_width=0, head_length=0, lw=2.25)

for i in range(0, 100):
    if i == 0:
        continue
    axs[0].axhline(0+0.2*i, color='grey', linestyle='--')
    axs[1].axhline(0+0.1*i, color='grey', linestyle='--')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6.5, 4, forward=True)
plt.show()
fig.savefig('tradeoff_new.pdf', bbox_inches='tight')
fig.savefig('tradeoff_new.png', bbox_inches='tight')
fig.savefig('tradeoff_new.svg', bbox_inches='tight')