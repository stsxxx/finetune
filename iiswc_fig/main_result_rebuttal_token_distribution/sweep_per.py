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
        kernels.append(ln_split[0])
        space.append(float(ln_split[1]))
        WIN2.append(float(ln_split[2]))
        WIN3.append(float(ln_split[3]))
        WIN4.append(float(ln_split[4]))
        WIN5.append(float(ln_split[5]))
        WIN6.append(float(ln_split[6]))
        WIN7.append(float(ln_split[7]))
        WIN8.append(float(ln_split[8]))
        
ind = numpy.arange(len(kernels))*1.75
# ind[-1] += 0.4
ax1 = plt.gca()
width = 0.125


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[0].bar(ind, space, width,  align='center',
              color='#d53e4f', edgecolor="black", label="Expert 0")
p0 = axs[0].bar(ind+width, WIN2, width,  align='center',
             color='#fdae61', edgecolor="black", label="Expert 1")
p0 = axs[0].bar(ind+2*width, WIN3, width,  align='center',
             color='#fee08b', edgecolor="black", label="Expert 2")
p0 = axs[0].bar(ind+3*width, WIN4, width,  align='center',
             color='#ffffbf', edgecolor="black", label="Expert 3")
p1 = axs[0].bar(ind+4*width, WIN5, width, align='center',
             color='#e6f598', edgecolor="black", label="Expert 4")
p1 = axs[0].bar(ind+5*width, WIN6, width, align='center',
             color='#abdda4', edgecolor="black", label="Expert 5")
p1 = axs[0].bar(ind+6*width, WIN7, width, align='center',
             color='#66c2a5', edgecolor="black", label="Expert 6")
p1 = axs[0].bar(ind+7*width, WIN8, width, align='center',
             color='#3288bd', edgecolor="black", label="Expert 7")


axs[0].set_xticks(ind+3.5*width) #, kernels, fontsize=graph_font_size-5, rotation=0)
# axs[0].tick_params(fontsize=graph_font_size-5, rotation=0)
axs[0].xaxis.set_tick_params(labelsize=graph_font_size-5, rotation=0)
axs[0].set_xticklabels(kernels)

axs[0].set_xlim(-0.5, len(kernels) * 1.75 - 0.45)
axs[0].set_ylim(0, 125)
for label in axs[0].get_yticklabels():
    label.set_fontproperties(font0)
# axs[0].set_xticklabels(ind+2*width, fontsize=graph_font_size-5, rotation=0)
axs[0].legend(fontsize=graph_font_size-7, ncol=4, loc="lower center", bbox_to_anchor=(0.48, 0.965))

axs[0].set_yticks(range(0, 125, 25))

for i in range(0, 10):
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
        WIN6.append(float(ln_split[6]))
        WIN7.append(float(ln_split[7]))
        WIN8.append(float(ln_split[8]))

ind = numpy.arange(len(kernels))*1.75
# ind[-1] += 0.4

p00 = plt.bar(ind, space, width,  align='center',
              color='#d53e4f', edgecolor="black", label="Expert 0")
p0 = plt.bar(ind+width, WIN2, width,  align='center',
             color='#fdae61', edgecolor="black", label="Expert 1")
p0 = plt.bar(ind+2*width, WIN3, width,  align='center',
             color='#fee08b', edgecolor="black", label="Expert 2")
p0 = plt.bar(ind+3*width, WIN4, width,  align='center',
             color='#ffffbf', edgecolor="black", label="Expert 3")
p1 = plt.bar(ind+4*width, WIN5, width, align='center',
             color='#e6f598', edgecolor="black", label="Expert 4")
p1 = plt.bar(ind+5*width, WIN6, width, align='center',
             color='#abdda4', edgecolor="black", label="Expert 5")
p1 = plt.bar(ind+6*width, WIN7, width, align='center',
             color='#66c2a5', edgecolor="black", label="Expert 6")
p1 = plt.bar(ind+7*width, WIN8, width, align='center',
             color='#3288bd', edgecolor="black", label="Expert 7")


plt.xticks(ind+3.5*width, kernels, fontsize=graph_font_size-5, rotation=0)

# plt.legend(fontsize=graph_font_size-7, ncol=4, loc="lower center", bbox_to_anchor=(0.48, 0.965))

ax1.yaxis.set_ticks(np.arange(0, 100, 25))
for i in range(0, 10):
    if i == 0:
        continue
    ax1.axhline(0+25*i, color='grey', linestyle='--')
ax1.set_ylim(0, 125)
for axis in [ax1.yaxis]:
    axis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax1.set_xlim(-0.5, len(kernels) * 1.75 - 0.45)

# plt.xlabel("Kernels", fontsize=graph_font_size-8, font_properties=font0)
plt.ylabel("                          Avg Num. of Token Per Query", fontsize=graph_font_size-8,
           font_properties=font0)

fig.text(0.132,1.16,"Mixtral", weight='bold', va='center', fontsize=graph_font_size-2)
fig.text(0.132,0.5375,"Mamba", weight='bold', va='center', fontsize=graph_font_size-2)

fig.text(0.18,1,"var=55.5", weight='normal', va='center', fontsize=graph_font_size-4)
fig.text(0.365,1.03,"var=112.3", weight='normal', va='center', fontsize=graph_font_size-4)
fig.text(0.565,1.09,"var=21.2", weight='normal', va='center', fontsize=graph_font_size-4)
fig.text(0.76,1.137,"var=79.2", weight='normal', va='center', fontsize=graph_font_size-4)

fig.text(0.18,1 - 0.6,"var=150.7", weight='normal', va='center', fontsize=graph_font_size-4)
fig.text(0.37,1.03 - 0.63,"var=93.3", weight='normal', va='center', fontsize=graph_font_size-4)
fig.text(0.565,1.09-0.62,"var=186.5", weight='normal', va='center', fontsize=graph_font_size-4)
fig.text(0.76,1.137-0.645,"var=187.9", weight='normal', va='center', fontsize=graph_font_size-4)



plt.yticks(range(0, 125, 25), fontsize=graph_font_size)
fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(20, 4, forward=True)
fig.set_size_inches(15, 5, forward=True)
plt.tight_layout()
fig.savefig('sweep_per.pdf',bbox_inches='tight')
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.svg',bbox_inches='tight')
# plt.show()
