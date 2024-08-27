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
plt.subplots_adjust(top=1.2, wspace=0.12, hspace=-0.02)

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
        WIN5.append(float(ln_split[5]))
        
ind = numpy.arange(len(kernels))*1.75
ind[-1] += 0.4
ax1 = plt.gca()
width = 0.25


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[0].bar(ind, space, width,  align='center',
              color='#b2182b', edgecolor="black", label="Dense(bsz=1)")
p0 = axs[0].bar(ind+width, WIN2, width,  align='center',
             color='#f4a582', edgecolor="black", label="Dense(bsz=10)")
p0 = axs[0].bar(ind+2*width, WIN3, width,  align='center',
             color='#d1e5f0', edgecolor="black", label="Sparse(bsz=1)")
p0 = axs[0].bar(ind+3*width, WIN4, width,  align='center',
             color='#4393c3', edgecolor="black", label="Sparse(bsz=10)")
p1 = axs[0].bar(ind+4*width, WIN5, width, align='center',
             color='#2166ac', edgecolor="black", label="Sparse(bsz=32)")


axs[0].set_xticks(ind+2*width) #, kernels, fontsize=graph_font_size-5, rotation=45)
# axs[0].tick_params(fontsize=graph_font_size-5, rotation=45)
axs[0].xaxis.set_tick_params(labelsize=graph_font_size-9, rotation=15)
axs[0].set_xticklabels(kernels)

axs[0].set_xlim(-0.5, len(kernels) * 1.75 + 0.05)
axs[0].set_ylim(0, 125)
for label in axs[0].get_yticklabels():
    label.set_fontproperties(font0)
# axs[0].set_xticklabels(ind+2*width, fontsize=graph_font_size-5, rotation=45)
axs[0].legend(fontsize=graph_font_size-10, ncol=3, loc="lower center", bbox_to_anchor=(0.48, 0.965))

axs[0].set_yticks(range(0, 120, 25))
axs[0].yaxis.set_tick_params(labelsize=graph_font_size-4)

for i in range(0, 5):
    if i == 0:
        continue
    axs[0].axhline(0+25*i, color='grey', linestyle='--')

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
        WIN4.append(float(ln_split[4]))
        WIN5.append(float(ln_split[5]))
ind = numpy.arange(len(kernels))*1.75
ind[-1] += 0.4

p00 = plt.bar(ind, space, width,  align='center',
              color='#b2182b', edgecolor="black", label="Dense(bsz=1)")
p0 = plt.bar(ind+width, WIN2, width,  align='center',
             color='#f4a582', edgecolor="black", label="Dense(bsz=30)")
p0 = plt.bar(ind+2*width, WIN3, width,  align='center',
             color='#d1e5f0', edgecolor="black", label="Sparse(bsz=1)")
p0 = plt.bar(ind+3*width, WIN4, width,  align='center',
             color='#4393c3', edgecolor="black", label="Sparse(bsz=30)")
p1 = plt.bar(ind+4*width, WIN5, width, align='center',
             color='#2166ac', edgecolor="black", label="Sparse(bsz=84)")

plt.xticks(ind+2*width, kernels, fontsize=graph_font_size-9, rotation=15)

plt.legend(fontsize=graph_font_size-10, ncol=3, loc="lower center", bbox_to_anchor=(0.48, 0.965))

ax1.yaxis.set_ticks(np.arange(0, 100, 25))
for i in range(0, 10):
    if i == 0:
        continue
    ax1.axhline(0+25*i, color='grey', linestyle='--')
ax1.set_ylim(0, 125)
for axis in [ax1.yaxis]:
    axis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax1.set_xlim(-0.5, len(kernels) * 1.75 + 0.05)


fig.text(0.058,0.85,"Mixtral", weight='bold', va='center', fontsize=graph_font_size-8)
fig.text(0.058,0.367,"Mamba", weight='bold', va='center', fontsize=graph_font_size-8)


# plt.xlabel("Kernels", fontsize=graph_font_size-8, font_properties=font0)
fig.text(-0.01, 0.485, '      SM Utilization (%)', fontsize=23, va='center', rotation='vertical')

# plt.ylabel("                               SM Utilization (%)", fontsize=graph_font_size-8,
#            font_properties=font0)
plt.yticks(range(0, 120, 25), fontsize=graph_font_size-4)
fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(20, 4, forward=True)
fig.set_size_inches(15, 6.5, forward=True)
plt.tight_layout()
fig.savefig('sweep_per.pdf',bbox_inches='tight')
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.svg',bbox_inches='tight')
# plt.show()
