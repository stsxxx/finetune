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
plt.subplots_adjust(top=1.2, wspace=0.12, hspace=-0.22)

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
WIN9 = []
WIN10 = []

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
        WIN5.append(float(ln_split[5]))
        WIN6.append(float(ln_split[6]))
        WIN7.append(float(ln_split[7]))
        WIN8.append(float(ln_split[8]))
        WIN9.append(float(ln_split[9]))
        WIN10.append(float(ln_split[10]))
        
ind = numpy.arange(len(kernels))*1.25
# ind[-1] += 0.4
ax1 = plt.gca()
width = 0.5


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[0].bar(ind+width, space, width,  align='center',
              color='#b2182b', edgecolor="black", label="matmul(w2)")
p0 = axs[0].bar(ind+width, WIN2, width,  align='center',
             color='#d6604d', edgecolor="black", bottom=space, label="w2_dequant")
tmp = np.array(space)
tmp += np.array(WIN2)
p0 = axs[0].bar(ind+width, WIN3, width,  align='center',
             color='#f4a582', edgecolor="black", bottom=tmp, label="matmul(w3)")
tmp += np.array(WIN3)
p0 = axs[0].bar(ind+1*width, WIN4, width,  align='center',
             color='#fddbc7', edgecolor="black", bottom=tmp, label="w3_dequant")
tmp += np.array(WIN4)
p0 = axs[0].bar(ind+1*width, WIN5, width,  align='center',
             color='#f7f7f7', edgecolor="black", bottom=tmp, label="matmul(w1)")
tmp += np.array(WIN5)
p0 = axs[0].bar(ind+1*width, WIN6, width,  align='center',
             color='#d1e5f0', edgecolor="black", bottom=tmp, label="w1_dequant")
tmp += np.array(WIN6)
p0 = axs[0].bar(ind+1*width, WIN7, width,  align='center',
             color='#92c5de', edgecolor="black", bottom=tmp, label="softmax")
tmp += np.array(WIN7)
p0 = axs[0].bar(ind+1*width, WIN8, width,  align='center',
             color='#4393c3', edgecolor="black", bottom=tmp, label="topk")
tmp += np.array(WIN8)
p0 = axs[0].bar(ind+1*width, WIN9, width,  align='center',
             color='#2166ac', edgecolor="black", bottom=tmp, label="matmul(router)")
tmp += np.array(WIN9)
p0 = axs[0].bar(ind+1*width, WIN10, width,  align='center',
             color='#66c2a5', edgecolor="black", bottom=tmp, label="router_dequant")
# p1 = plt.bar(ind+4*width, WIN5, width, align='center',
#              color='#2166ac', edgecolor="black", label="Sparse(bsz=16)")

for i in range(0, 4):
    if i == 0:
        continue
    axs[0].axhline(0+2000*i, color='grey', linestyle='--')

axs[0].set_xticks(ind+1*width) #, kernels, fontsize=graph_font_size-5, rotation=0)
# axs[0].tick_params(fontsize=graph_font_size-5, rotation=0)
axs[0].xaxis.set_tick_params(labelsize=graph_font_size-2, rotation=0)
axs[0].set_xticklabels(kernels)
axs[0].legend(fontsize=graph_font_size-5, ncol=4, loc="lower center", bbox_to_anchor=(0.48, 0.965))
# axs[0].set_ylim(0, 6000)
axs[0].set_yticks(range(0, 6100, 2000))
axs[0].yaxis.set_tick_params(labelsize=graph_font_size-3, rotation=0)


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

ind = numpy.arange(len(kernels))*1.25
# ind[-1] += 0.4
ax1 = plt.gca()
width = 0.5


font0 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)

p00 = axs[1].bar(ind+width, space, width,  align='center',
              color='#b2182b', edgecolor="black", label="matmul(w1)")
p0 = axs[1].bar(ind+width, WIN2, width,  align='center',
             color='#d6604d', edgecolor="black", bottom=space, label="gelu")
tmp = np.array(space)
tmp += np.array(WIN2)
p0 = axs[1].bar(ind+width, WIN3, width,  align='center',
             color='#f4a582', edgecolor="black", bottom=tmp, label="matmul(w2)")
tmp += np.array(WIN3)
p0 = axs[1].bar(ind+1*width, WIN4, width,  align='center',
             color='#fddbc7', edgecolor="black", bottom=tmp, label="elementwise_mult")
tmp += np.array(WIN4)
p0 = axs[1].bar(ind+1*width, WIN5, width,  align='center',
             color='#f7f7f7', edgecolor="black", bottom=tmp, label="top_k")
tmp += np.array(WIN5)
p0 = axs[1].bar(ind+1*width, WIN6, width,  align='center',
             color='#d1e5f0', edgecolor="black", bottom=tmp, label="sigmoid")
tmp += np.array(WIN6)
p0 = axs[1].bar(ind+1*width, WIN7, width,  align='center',
             color='#92c5de', edgecolor="black", bottom=tmp, label="matmul(router)")

axs[1].set_xticks(ind+1*width) #, kernels, fontsize=graph_font_size-5, rotation=0)
# axs[1].tick_params(fontsize=graph_font_size-5, rotation=0)
axs[1].xaxis.set_tick_params(labelsize=graph_font_size-2, rotation=0)
axs[1].set_xticklabels(kernels)
axs[1].legend(fontsize=graph_font_size-5, ncol=4, loc="lower center", bbox_to_anchor=(0.48, 0.965))

axs[1].yaxis.set_tick_params(labelsize=graph_font_size-3, rotation=0)


# plt.set_xticklabels(ind+2*width, fontsize=graph_font_size-5, rotation=0)




plt.xticks(ind+1*width, kernels, fontsize=graph_font_size-5, rotation=0)



for i in range(0, 6):
    if i == 0:
        continue
    ax1.axhline(0+400*i, color='grey', linestyle='--')
# ax1.set_ylim(0, 2000)
for axis in [ax1.yaxis]:
    axis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
# ax1.set_xlim(-0.5, len(kernels) * 1.26)

fig.text(-0.015,0.47,"    Execution Time Breakdown (μs)", va='center', rotation='vertical', fontsize=graph_font_size)

fig.text(0.072,0.765,"Mixtral", weight='bold', va='center', fontsize=graph_font_size-2)
fig.text(0.072,0.3075,"Mamba", weight='bold', va='center', fontsize=graph_font_size-2)

# plt.xlabel("Kernels", fontsize=graph_font_size-8, font_properties=font0)
# plt.ylabel("                              Latency Breakdown (seconds)", fontsize=graph_font_size-8,
#            font_properties=font0)
# plt.yticks([0, 0.5, 1, 1.5], fontsize=graph_font_size)
axs[1].set_yticks(range(0, 2100, 400))
fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(20, 4, forward=True)
fig.set_size_inches(15, 7.5, forward=True)
plt.tight_layout()
fig.savefig('sweep_per.pdf',bbox_inches='tight')
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.svg',bbox_inches='tight')
# plt.show()
