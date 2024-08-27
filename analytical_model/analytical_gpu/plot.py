import numpy, scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize

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


fig, axs = plt.subplots(1, 3)
plt.subplots_adjust(top=1.2, wspace=0.25, hspace=0.22)

coeff=1.8

x = numpy.array([1, 1, 2, 3])
sparsity = numpy.array([1, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/sparsity[i]/coeff) for i in range(len(x))])
y = numpy.array([0.3886286631, 0.4490951743,	0.8150684641,	1.1513719])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
axs[0].plot(x[:1], y[:1], 'o', markersize=12, color='#d73027')
axs[0].plot(x[1:], y[1:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 5, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[0].plot(xx, yy1, '-', lw=5, color='#d73027', label='Dense')
axs[0].plot(xx, yy2, '-', lw=5, color='#4575b4', label='Sparse')
axs[0].set_title(f"Mixtral-CS-A100-40GB", fontsize=25)
# axs[0].set_xlabel('Batch size')
# axs[0].set_ylabel('Throughput (queries/sec)')
axs[0].set_ylim([0, None])
# plt.savefig('mixtral_cs.png')

axs[0].legend(fontsize=graph_font_size, ncol=2, loc="lower center", bbox_to_anchor=(1.5+0.2, 0.103+0.965))

rms = 0.0
for i in range(0, len(y)):
    if i < 1:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[0].text(-0.125, 1.43, f'RMSE={rms:.2f}', fontsize=25)
    


coeff=1.8

x = numpy.array([1, 2, 3, 4, 5, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17])
sparsity = numpy.array([1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/sparsity[i]/coeff) for i in range(len(x))])
y = numpy.array([0.4153268396,	0.6734088813,	0.9212998066,	0.9212998066,	0.9531133065, 0.4736763151,	0.8774646886,	1.1201166,	1.708054828,	1.800397168,	2.031587942,	2.189323022,	2.282276004,	2.305001484,	2.567787669,	2.589650462,	2.678787645,	2.693921973,	2.739929458])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
axs[1].plot(x[:5], y[:5], 'o', markersize=12, color='#d73027')
axs[1].plot(x[5:], y[5:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 20, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[1].plot(xx, yy1, '-', lw=5, color='#d73027')
axs[1].plot(xx, yy2, '-', lw=5, color='#4575b4')
axs[1].set_title(f"Mixtral-CS-A100-80GB", fontsize=25)
# axs[1].set_xlabel('Batch size')
# axs[1].set_ylabel('Throughput (queries/sec)')
axs[1].set_ylim([0, None])
# plt.savefig('mamba_cs.png')


rms = 0.0
for i in range(0, len(y)):
    if i < 5:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[1].text(-0.18, 2.73, f'RMSE={rms:.2f}', fontsize=25)

coeff=1.8

x = numpy.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17])
sparsity = numpy.array([1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/sparsity[i]/coeff) for i in range(len(x))])
y = numpy.array([0.5468527884,	0.9480089158,	1.27383748,	1.501140867,	1.656801793, 0.6716115184,	1.277928978,	1.491977043,	1.79810839,	2.167855772,	2.634467155,	2.936952442,	3.152775956,	4.135249112,	4.311939502,	4.394074678,	4.666207512,	4.635430356,	4.89877178])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
axs[2].plot(x[:5], y[:5], 'o', markersize=12, color='#d73027')
axs[2].plot(x[5:], y[5:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 20, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[2].plot(xx, yy1, '-', lw=5, color='#d73027')
axs[2].plot(xx, yy2, '-', lw=5, color='#4575b4')
axs[2].set_title(f"Mixtral-CS-H100", fontsize=25)
# axs[2].set_xlabel('Batch size')
# axs[2].set_ylabel('Throughput (queries/sec)')
axs[2].set_ylim([0, None])
# plt.savefig('mixtral_math.png')

rms = 0.0
for i in range(0, len(y)):
    if i < 1:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[2].text(-0.03, 4.71, f'RMSE={rms:.2f}', fontsize=25)




for i in range(0, 3):
    for label in axs[i].get_yticklabels():
        label.set_fontproperties(font1)
    for label in axs[i].get_xticklabels():
        label.set_fontproperties(font1)


# axs[2].set_xlabel('Batch size')
# axs[2].set_ylabel('Throughput (queries/sec)')
fig.text(0.05, 0.06, "Throughput (queries/sec)", fontsize=25, rotation=90)
fig.text(0.45, -0.12, "Batch size", fontsize=25, rotation=0)


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 3., forward=True)
fig.savefig('./analytical_gpu/throughput_gpus.png',bbox_inches='tight')
fig.savefig('./analytical_gpu/throughput_gpus.pdf',bbox_inches='tight')


