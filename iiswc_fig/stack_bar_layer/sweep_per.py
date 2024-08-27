import sys
import re
import numpy
import os
import math
import matplotlib
import matplotlib.pyplot as plt 
import glob
import argparse
from pylab import *

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


fig, axs = plt.subplots(2, 1)
fig.set_size_inches(10.5, 2.5)
plt.subplots_adjust(top=1.2, wspace=0.12, hspace=-0.52)

# fig, axs = plt.subplots(2, 1)
# fig.set_size_inches(10.5, 2.5)
# plt.subplots_adjust(top=1.2, wspace=0.12, hspace=0.02)

space = []
WIN2 = []
WIN3 = []
WIN4 = []
WIN5 = []
WIN6 = []
WIN7 = []
WIN8 = []

kernels = []

file = "sweep_per.txt"
with open(file) as fp:
    # print("Reading from file", file)
    for ln in fp:
        if "\n" in ln:
            ln = ln[:-1]
        ln_split = ln.split("\t")
        print(ln_split)
        kernels.append(ln_split[0])
        space.append(float(ln_split[1]))
        WIN2.append(float(ln_split[2]))
        WIN3.append(float(ln_split[3]))
        WIN4.append(float(ln_split[4]))
        # WIN5.append(float(ln_split[5]))
        
ind = numpy.arange(len(kernels))*1.25
# ind[-1] += 0.4
ax1 = plt.gca()
width = 0.5


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[0].bar(ind+width, space, width,  align='center',
              color='#b2182b', edgecolor="black", label="Input normalization")
p0 = axs[0].bar(ind+width, WIN2, width,  align='center',
             color='#f4a582', edgecolor="black", bottom=space, label="Attention")
p0 = axs[0].bar(ind+width, WIN3, width,  align='center',
             color='#d1e5f0', edgecolor="black", bottom=np.array(space)+np.array(WIN2), label="Post attention norm.")
p0 = axs[0].bar(ind+1*width, WIN4, width,  align='center',
             color='#4393c3', edgecolor="black", bottom=np.array(space)+np.array(WIN2)+np.array(WIN3), label="MoE")
# p1 = plt.bar(ind+4*width, WIN5, width, align='center',
#              color='#2166ac', edgecolor="black", label="Sparse(bsz=16)")

for i in range(0, 6):
    if i == 0:
        continue
    axs[0].axhline(0+1*i, color='grey', linestyle='--')

axs[0].set_xticks(ind+1*width) #, kernels, fontsize=graph_font_size-5, rotation=0)
# axs[0].tick_params(fontsize=graph_font_size-5, rotation=0)
axs[0].xaxis.set_tick_params(labelsize=graph_font_size-2, rotation=0)
axs[0].set_xticklabels(kernels)
axs[0].legend(fontsize=graph_font_size-5, ncol=4, loc="lower center", bbox_to_anchor=(0.48, 0.965))
axs[0].set_ylim(0, 5.2)
axs[0].set_yticks(range(0, 6, 1))
axs[0].yaxis.set_tick_params(labelsize=graph_font_size-2, rotation=0)


import matplotlib.ticker as tkr
fmtr = tkr.StrMethodFormatter('{x:.1f}')
axs[0].yaxis.set_major_formatter(fmtr)


space = []
WIN2 = []
WIN3 = []
WIN4 = []
WIN5 = []
WIN6 = []
WIN7 = []
WIN8 = []

kernels = []

file = "sweep_per_256.txt"
with open(file) as fp:
    # print("Reading from file", file)
    for ln in fp:
        if "\n" in ln:
            ln = ln[:-1]
        ln_split = ln.split("\t")
        print(ln_split)
        kernels.append(ln_split[0])
        space.append(float(ln_split[1]))
        WIN2.append(float(ln_split[2]))
        WIN3.append(float(ln_split[3]))
        # WIN4.append(float(ln_split[4]))
        # WIN5.append(float(ln_split[5]))
        
ind = numpy.arange(len(kernels))*1.25
# ind[-1] += 0.4
ax1 = plt.gca()
width = 0.5


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[1].bar(ind+width, WIN3, width,  align='center',
              color='#33a02c', edgecolor="black", label="RMS layernorm")
p0 = axs[1].bar(ind+width, WIN2, width,  align='center',
             color='#b2df8a', edgecolor="black", bottom=WIN3, label="Mamba")
p0 = axs[1].bar(ind+width, space, width,  align='center',
             color='#4393c3', edgecolor="black", bottom=np.array(WIN3)+np.array(WIN2), label="MoE")
# p0 = plt.bar(ind+3*width, WIN4, width,  align='center',
#              color='#4393c3', edgecolor="black", label="Sparse(bsz=4)")
# p1 = plt.bar(ind+4*width, WIN5, width, align='center',
#              color='#2166ac', edgecolor="black", label="Sparse(bsz=16)")
axs[1].set_ylim(0, 1.56)

axs[1].set_xticks(ind+1*width) #, kernels, fontsize=graph_font_size-5, rotation=0)
# axs[1].tick_params(fontsize=graph_font_size-5, rotation=0)
axs[1].xaxis.set_tick_params(labelsize=graph_font_size-2, rotation=0)
axs[1].set_xticklabels(kernels)
axs[1].legend(fontsize=graph_font_size-5, ncol=3, loc="lower center", bbox_to_anchor=(0.48, 0.965))

import matplotlib.ticker as tkr
fmtr = tkr.StrMethodFormatter('{x:.1f}')
axs[1].yaxis.set_major_formatter(fmtr)

axs[1].yaxis.set_tick_params(labelsize=graph_font_size-2, rotation=0)


# plt.set_xticklabels(ind+2*width, fontsize=graph_font_size-5, rotation=0)




plt.xticks(ind+1*width, kernels, fontsize=graph_font_size-5, rotation=0)


for i in range(0, 10):
    if i == 0:
        continue
    ax1.axhline(0+0.5*i, color='grey', linestyle='--')
# ax1.set_ylim(0, 1.75)
for axis in [ax1.yaxis]:
    axis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
# ax1.set_xlim(-0.5, len(kernels) * 1.26)

# plt.xlabel("Kernels", fontsize=graph_font_size-8, font_properties=font0)
# plt.ylabel("                              Latency Breakdown (seconds)", fontsize=graph_font_size-8,
#            font_properties=font0)
fig.text(-0.04,0.5,"    Execution Time\nBreakdown (seconds)", va='center', rotation='vertical', fontsize=graph_font_size)

fig.text(0.058,0.8825,"Mixtral", weight='bold', va='center', fontsize=graph_font_size-2)
fig.text(0.058,0.38,"Mamba", weight='bold', va='center', fontsize=graph_font_size-2)


# plt.yticks([0, 0.5, 1, 1.5], fontsize=graph_font_size)
plt.yticks([0, 0.5, 1, 1.5])
fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(20, 4, forward=True)
fig.set_size_inches(15, 7.5, forward=True)
plt.tight_layout()
fig.savefig('sweep_per.pdf',bbox_inches='tight')
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.svg',bbox_inches='tight')
# plt.show()
