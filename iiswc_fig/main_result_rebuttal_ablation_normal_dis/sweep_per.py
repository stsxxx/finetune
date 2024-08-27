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


space = []
WIN2 = []
WIN3 = []
WIN4 = []
WIN5 = []
WIN6 = []
WIN7 = []
WIN8 = []

kernels = []


fig, axs = plt.subplots(2, 1)
fig.set_size_inches(10.5, 2.5)
plt.subplots_adjust(top=1.2, wspace=0.12, hspace=0.3)

file = "sweep_per.txt"
with open(file) as fp:
    # print("Reading from file", file)
    for ln in fp:
        if "\n" in ln:
            ln = ln[:-1]
        ln_split = ln.split("\t")
        print(ln_split)
        kernels.append(float(ln_split[0]))
        space.append(float(ln_split[1]))

        if float(ln_split[0]) > 400:
            break
        
width = 0.1


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[0].bar(kernels, space, width,  align='center',
              color='#000000', edgecolor="black", label="SeqLen=64")

axs[0].set_xlim(0, 400)
for label in axs[0].get_yticklabels():
    label.set_fontproperties(font0)
for label in axs[0].get_xticklabels():
    label.set_fontproperties(font0)


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
        kernels.append(float(ln_split[0]))
        space.append(float(ln_split[1]))

        if float(ln_split[0]) > 400:
            break
        
width = 0.1


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[1].bar(kernels, space, width,  align='center',
              color='#000000', edgecolor="black", label="SeqLen=64")

axs[1].set_xlim(0, 400)
for label in axs[1].get_yticklabels():
    label.set_fontproperties(font0)
for label in axs[1].get_xticklabels():
    label.set_fontproperties(font0)


fig.text(0.065,0.938,"CS", weight='bold', va='center', fontsize=graph_font_size-2)
fig.text(0.065,0.445,"MATH", weight='bold', va='center', fontsize=graph_font_size-2)

fig.text(0.244,0.938,"Median=79", color='red', weight='normal', va='center', fontsize=graph_font_size-2)
fig.text(0.46,0.445,"Median=174", color='red', weight='normal', va='center', fontsize=graph_font_size-2)

axs[0].axvline(79, color='red', linestyle='-')
axs[1].axvline(174, color='red', linestyle='-')

# plt.xlabel("Kernels", fontsize=graph_font_size-8, font_properties=font0)
fig.text(-0.015,0.38,"Frequency", rotation='vertical', fontsize=graph_font_size-2,
           font_properties=font0)
fig.text(0.42,-0.04,"Sequence Length", rotation='horizontal', fontsize=graph_font_size-2,
           font_properties=font0)

# plt.yticks(range(0, 105, 25), fontsize=graph_font_size)
fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(20, 4, forward=True)
fig.set_size_inches(15, 7, forward=True)
plt.tight_layout()
fig.savefig('sweep_per.pdf',bbox_inches='tight')
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.svg',bbox_inches='tight')
# plt.show()
