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


fig, axs = plt.subplots(2, 2)

# plt.subplots_adjust(top=1.5, wspace=0.12, hspace=10.5)

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
        
ind = numpy.arange(5)*1.75
# ind[-1] += 0.4
ax1 = plt.gca()
width = 0.75


font0 = matplotlib.font_manager.FontProperties()
font1 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font0.set_size(graph_font_size)
font1.set_size(graph_font_size - 9)

p00 = axs[0][0].bar(ind[0], [0.3213791846], width,  align='center',
              color='#b2182b', edgecolor="black", label="Dense(bsz=1)")
p00 = axs[0][0].bar(ind[1], [0.5138618605], width,  align='center',
              color='#f4a582', edgecolor="black", label="Dense(bsz=2)")
p00 = axs[0][0].bar(ind[2], [0.341252845], width,  align='center',
              color='#d1e5f0', edgecolor="black", label="Sparse(bsz=1)")
p00 = axs[0][0].bar(ind[3], [0.6567373669], width,  align='center',
              color='#92c5de', edgecolor="black", label="Sparse(bsz=2)")
p00 = axs[0][0].bar(ind[4], [1.655243555], width,  align='center',
              color='#2166ac', edgecolor="black", label="Sparse(bsz=8)")
axs[0][0].set_xlabel("Mixtral-CS", fontsize=graph_font_size-7)
# axs[0][0].set_ylabel("Throughput\n(quries/second)", fontsize=graph_font_size-8,
#            font_properties=font0)

axs[0][0].text(ind[0] - 0.255, 0.3213791846 + 0.1, str(0.3), fontsize=18)
axs[0][0].text(ind[1] - 0.255, 0.5138618605 + 0.1, str(0.5), fontsize=18)
axs[0][0].text(ind[2] - 0.255, 0.341252845 + 0.1, str(0.3), fontsize=18)
axs[0][0].text(ind[3] - 0.255, 0.6567373669 + 0.05, str(0.7), fontsize=18)
axs[0][0].text(ind[4] - 0.255, 1.655243555 + 0.05, str(1.7), fontsize=18)

for i in range(0, 5):
    if i == 0:
        continue
    axs[0][0].axhline(0+0.5*i, color='grey', linestyle='--')
axs[0][0].xaxis.set_ticks(ind+0*width)
axs[0][0].tick_params(labelbottom=False)

axs[0][0].yaxis.set_ticks(np.arange(0, 2.5, 0.5))
axs[0][0].set_xlim(-0.5-0.25, 5 * 1.75-1)
for label in axs[0][0].get_yticklabels():
    label.set_fontproperties(font1)
axs[0][0].legend(fontsize=graph_font_size-11, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 0.002+0.965))

##########


p00 = axs[1][0].bar(ind[0], [2.324241983], width,  align='center',
              color='#b2182b', edgecolor="black", label="Dense(bsz=1)")
p00 = axs[1][0].bar(ind[1], [7.877675456], width,  align='center',
              color='#f4a582', edgecolor="black", label="Dense(bsz=6)")
p00 = axs[1][0].bar(ind[2], [2.40759297], width,  align='center',
              color='#d1e5f0', edgecolor="black", label="Sparse(bsz=1)")
p00 = axs[1][0].bar(ind[3], [10.46236322], width,  align='center',
              color='#92c5de', edgecolor="black", label="Sparse(bsz=6)")
p00 = axs[1][0].bar(ind[4], [14.87415719], width,  align='center',
              color='#2166ac', edgecolor="black", label="Sparse(bsz=20)")
axs[1][0].set_xlabel("Blackmamba-CS", fontsize=graph_font_size-7)
# axs[1][0].set_ylabel("Throughput\n(quries/second)", fontsize=graph_font_size-8,
#            font_properties=font0)

axs[1][0].text(ind[0] - 0.255, 2.324241983 + 0.6, str(2.3), fontsize=18)
axs[1][0].text(ind[1] - 0.255, 7.877675456 + 0.6, str(7.9), fontsize=18)
axs[1][0].text(ind[2] - 0.255, 2.40759297 + 0.6, str(2.4), fontsize=18)
axs[1][0].text(ind[3] - 0.335, 10.46236322 + 0.6, str(10.5), fontsize=18)
axs[1][0].text(ind[4] - 0.335, 14.87415719 + 0.6, str(14.9), fontsize=18)

for i in range(0, 5):
    if i == 0:
        continue
    axs[1][0].axhline(0+5*i, color='grey', linestyle='--')
axs[1][0].xaxis.set_ticks(ind+0*width)
axs[1][0].tick_params(labelbottom=False)

axs[1][0].yaxis.set_ticks(np.arange(0, 25, 5))
axs[1][0].set_xlim(-0.5-0.25, 5 * 1.75-1)
for label in axs[1][0].get_yticklabels():
    label.set_fontproperties(font1)
axs[1][0].legend(fontsize=graph_font_size-11, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 0.002+0.965))

##########


# p00 = axs[0][1].bar(ind[0], [1], width,  align='center',
#               color='#b2182b', edgecolor="black", label="Dense(bsz=1)")
p00 = axs[0][1].bar(ind[1], [0.2833587728], width,  align='center',
              color='#f4a582', edgecolor="black", label="Dense(bsz=1)")
p00 = axs[0][1].bar(ind[2], [0.3354076283], width,  align='center',
              color='#d1e5f0', edgecolor="black", label="Sparse(bsz=1)")
p00 = axs[0][1].bar(ind[3], [1.005476933], width,  align='center',
              color='#92c5de', edgecolor="black", label="Sparse(bsz=3)")
# p00 = axs[0][1].bar(ind[4], [3.548423518], width,  align='center',
#               color='#2166ac', edgecolor="black", label="Sparse(bsz=10)")
axs[0][1].set_xlabel("Mixtral-MATH", fontsize=graph_font_size-7)


axs[0][1].text(ind[1] - 0.255, 0.2833587728 + 0.1, str(0.3), fontsize=18)
axs[0][1].text(ind[2] - 0.255, 0.3354076283 + 0.1, str(0.3), fontsize=18)
axs[0][1].text(ind[3] - 0.255, 1.005476933 + 0.1, str(1.0), fontsize=18)

fig.text(-0.01, 0.43, '      Throughput (quries/second)', fontsize=20, va='center', rotation='vertical')
# plt.text(-6.5, 4.75, "Throughput\n(quries/second)", fontsize=20, rotation=90)


for i in range(0, 5):
    if i == 0:
        continue
    axs[0][1].axhline(0+0.5*i, color='grey', linestyle='--')

axs[0][1].xaxis.set_ticks(ind[1:4])
axs[0][1].tick_params(labelbottom=False)

axs[0][1].yaxis.set_ticks((np.arange(0, 2.5, 0.5)))
axs[0][1].set_xlim(-0.5-0.25, 5 * 1.75-1)
for label in axs[0][1].get_yticklabels():
    label.set_fontproperties(font1)
axs[0][1].legend(fontsize=graph_font_size-11, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 0.002+0.965))

##########


p00 = axs[1][1].bar(ind[0], [2.218754555], width,  align='center',
              color='#b2182b', edgecolor="black", label="Dense(bsz=1)")
p00 = axs[1][1].bar(ind[1], [5.321407704], width,  align='center',
              color='#f4a582', edgecolor="black", label="Dense(bsz=2)")
p00 = axs[1][1].bar(ind[2], [2.173860115], width,  align='center',
              color='#d1e5f0', edgecolor="black", label="Sparse(bsz=1)")
p00 = axs[1][1].bar(ind[3], [6.453015914], width,  align='center',
              color='#92c5de', edgecolor="black", label="Sparse(bsz=2)")
p00 = axs[1][1].bar(ind[4], [11.6119473], width,  align='center',
              color='#2166ac', edgecolor="black", label="Sparse(bsz=8)")
axs[1][1].set_xlabel("Blackmamba-MATH", fontsize=graph_font_size-7)
# axs[1][1].set_ylabel("Throughput\n(quries/second)", fontsize=graph_font_size-8,
#            font_properties=font0)


for i in range(0, 5):
    if i == 0:
        continue
    axs[1][1].axhline(0+5*i, color='grey', linestyle='--')
axs[1][1].xaxis.set_ticks(ind+0*width)
axs[1][1].tick_params(labelbottom=False)

axs[1][1].yaxis.set_ticks(np.arange(0, 25, 5))
axs[1][1].set_xlim(-0.5-0.25, 5 * 1.75-1)
for label in axs[1][1].get_yticklabels():
    label.set_fontproperties(font1)
axs[1][1].legend(fontsize=graph_font_size-11, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 0.002+0.965))


axs[1][1].text(ind[0] - 0.255, 2.218754555 + 0.6, str(2.2), fontsize=18)
axs[1][1].text(ind[1] - 0.255, 5.321407704 + 0.6, str(5.3), fontsize=18)
axs[1][1].text(ind[2] - 0.255, 2.173860115 + 0.6, str(2.2), fontsize=18)
axs[1][1].text(ind[3] - 0.255, 6.453015914 + 0.6, str(6.5), fontsize=18)
axs[1][1].text(ind[4] - 0.335, 11.6119473 + 0.6, str(11.6), fontsize=18)

fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(20, 4, forward=True)
fig.set_size_inches(15, 5.5, forward=True)
plt.tight_layout()
fig.savefig('sweep_per.pdf',bbox_inches='tight')
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.svg',bbox_inches='tight')
# plt.show()
