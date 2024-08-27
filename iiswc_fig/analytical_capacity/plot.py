import numpy, scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.patches import Rectangle

GPU_dict = {
"H100":	[80,	12.29],
"A100":	[40,	4.097],
"V100":	[16,	3.06],
"K80":	[12,	0.9],
"L4":	[24,	0.8048],
"A10G":	[16,	1.006],
"T4":	[16,	0.526],
"M60":	[8,	0.75],
}

font1 = matplotlib.font_manager.FontProperties()
# font0.set_weight('bold')
graph_font_size = 25
font1.set_size(graph_font_size - 4)


fig, axs = plt.subplots(1, 1)
fig.set_size_inches(10.5, 2.5)
plt.subplots_adjust(top=1.2, wspace=0.12, hspace=0.22)

coeff=1.8
x = numpy.array([1, 2, 1, 2, 3, 4, 5, 6, 7, 8])
sparsity = numpy.array([1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/(sparsity[i]*coeff)) for i in range(len(x))])
y = numpy.array([0.3213791846, 0.5138618605, 0.341252845, 0.6567373669, 
                 0.9263685242,	1.134794525,	1.306288801,	1.460184208,	1.578898838, 1.655243555])

out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()

real_point_x = [40, 48, 80]
real_point_y = [6, 8, 27]

axs.plot(real_point_x, real_point_y, '^', markersize=24, color='#d73027', label='Ground Truth')

# axs.plot(x[:2], y[:2], 'o', markersize=12, color='#d73027')
# axs.plot(x[2:], y[2:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 128, 100)


C0 = 62
C1 = 0.89
model_mem = 23.35
seq_length = 512
sparsity = 0.25

yy1 = [C0*(xx[i]-model_mem)/(seq_length*((1-C1)+C1*sparsity)) for i in range(len(xx))]


# print(C0*(120-model_mem)/(seq_length*((1-C1)+C1*sparsity)))
# exit(0)
# yy1 = out[0][0]+out[0][1]*numpy.log(xx)
# yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs.plot(xx, yy1, '-', lw=5, color='#d73027', label='Projection')
# axs.plot(xx, yy2, '-', lw=5, color='#4575b4', label='A100')
# axs.set_title(f"Projected max batch size of Mixtral", fontsize=25)
# axs.set_xlabel('Batch size')
# axs.set_ylabel('Throughput (queries/sec)')
axs.set_xlim(0, 128)
axs.set_ylim([0, None])
# plt.savefig('mixtral_cs.png')

axs.legend(fontsize=graph_font_size, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 0.01+0.965))

# rms = 0.0
# for i in range(0, len(y)):
#     if i < 2:
#         rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
#     else:
#         rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
# import math
# rms = math.sqrt(rms / len(y))
# axs.text(-0.125, 1.7, f'RMSE={rms:.2f}', fontsize=25)
    
    


coeff=2.5

x = numpy.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
sparsity = numpy.array([1, 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/(sparsity[i]*coeff)) for i in range(len(x))])
y = numpy.array([2.324241983,	4.435789726,	5.921728955,	6.994841305,	7.768480633,	7.877675456,	2.40759297,	4.586554241,	6.299510213,	8.12722686,	9.305433163,	10.46236322,	11.17621879,	12.11379701,	12.59296758,	12.89725147,	13.31887791,	13.67491446,	13.82070674,	14.1110796,	14.02404563,	14.08680855,	14.40473858,	14.85546377,	14.853169,	14.87415719])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
# axs[1].plot(x[:6], y[:6], '^', markersize=12, color='#d73027')
# axs[1].plot(x[6:], y[6:], '^', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 22, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
# axs[1].plot(xx, yy1, '-', lw=5, color='#d73027')
# axs[1].plot(xx, yy2, '-', lw=5, color='#4575b4')
# axs[1].set_title(f"Mamba", fontsize=25)
# # axs[1].set_xlabel('Batch size')
# # axs[1].set_ylabel('Throughput (queries/sec)')
# axs[1].set_xlim(0, 128)

# plt.savefig('mamba_cs.png')


real_point_x = [40, 48, 80]
real_point_y = [6, 8, 27]

# axs[1].plot(real_point_x, real_point_y, '^', markersize=24, color='#d73027')

# axs.plot(x[:2], y[:2], 'o', markersize=12, color='#d73027')
# axs.plot(x[2:], y[2:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 128, 100)


C0 = 62
C1 = 0.89
model_mem = 23.35
seq_length = 512
sparsity = 0.25

yy1 = [C0*(xx[i]-model_mem)/(seq_length*((1-C1)+C1*sparsity)) for i in range(len(xx))]


# print(C0*(120-model_mem)/(seq_length*((1-C1)+C1*sparsity)))
# exit(0)
# yy1 = out[0][0]+out[0][1]*numpy.log(xx)
# yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
# axs[1].plot(xx, yy1, '-', lw=5, color='#d73027')
# # axs.plot(xx, yy2, '-', lw=5, color='#4575b4', label='A100')
# axs[1].set_title(f"Mixtral", fontsize=25)
# # axs.set_xlabel('Batch size')
# # axs.set_ylabel('Throughput (queries/sec)')
# axs[1].set_xlim(0, 128)
# axs[1].set_ylim([0, None])


# axs[1].text(32, 10, "A100", fontsize=25)
# axs[1].text(44, 13, "A40", fontsize=25)
# axs[1].text(69, 30, "H100", fontsize=25)


# rms = 0.0
# for i in range(0, len(y)):
#     if i < 6:
#         rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
#     else:
#         rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
# import math
# rms = math.sqrt(rms / len(y))
# axs[1].text(-0.18, 14.7, f'RMSE={rms:.2f}', fontsize=25)


for i in range(0, 1):
    for label in axs.get_yticklabels():
        label.set_fontproperties(font1)
    for label in axs.get_xticklabels():
        label.set_fontproperties(font1)


axs.text(24, 9, "A100-40GB", fontsize=25)
axs.text(57.5, 31.0, "A100-80GB", fontsize=25)
axs.text(44, 13, "A40", fontsize=25)
axs.text(67, 26.5, "H100", fontsize=25)

axs.text(87, 30, "bsz=28", fontsize=25)
axs.text(103, 35.5, "bsz=35", fontsize=25)

axs.axvline(100, ymin=0, ymax=0.695, lw=4, color='black', linestyle='--')
axs.axvline(120, ymin=0, ymax=0.87, lw=4, color='black', linestyle='--')

# axs[1].axvline(100, ymin=0, ymax=0.695, lw=4, color='black', linestyle='--')
# axs[1].axvline(120, ymin=0, ymax=0.87, lw=4, color='black', linestyle='--')


axs.add_patch(Rectangle((80, 0), 1000, 1000, facecolor="#d9d9d9"))
# axs[1].add_patch(Rectangle((80, 0), 1000, 1000, facecolor="#d9d9d9"))

axs.text(82, 1.7, "Projected GPU capacity", fontsize=25, bbox=dict(facecolor='white', edgecolor='black'))
# axs[1].text(82, 0.8, "Projected GPU capacity", fontsize=25, bbox=dict(facecolor='white', edgecolor='black'))


# axs[1].set_xlabel('Batch size')
# axs[1].set_ylabel('Throughput (queries/sec)')
fig.text(0.065, 0.3, "Max batch size", fontsize=27, rotation=90)
fig.text(0.4, -0.06, "GPU DRAM capacity", fontsize=25, rotation=0)


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 4., forward=True)
fig.savefig('sweep_per.png',bbox_inches='tight')
fig.savefig('sweep_per.pdf',bbox_inches='tight')


