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


fig, axs = plt.subplots(2, 2)
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
axs[0][0].plot(x[:2], y[:2], 'o', markersize=12, color='#d73027')
axs[0][0].plot(x[2:], y[2:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 10, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[0][0].plot(xx, yy1, '-', lw=5, color='#d73027', label='Dense')
axs[0][0].plot(xx, yy2, '-', lw=5, color='#4575b4', label='Sparse')
axs[0][0].set_title(f"Mixtral-CS", fontsize=25)
# axs[0][0].set_xlabel('Batch size')
# axs[0][0].set_ylabel('Throughput (queries/sec)')
axs[0][0].set_ylim([0, None])
# plt.savefig('mixtral_cs.png')

axs[0][0].legend(fontsize=graph_font_size, ncol=2, loc="lower center", bbox_to_anchor=(1.0, 0.12+0.965))

rms = 0.0
for i in range(0, len(y)):
    if i < 2:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[0][0].text(-0.125, 1.7, f'RMSE={rms:.2f}', fontsize=25)
    


coeff=2.5

x = numpy.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
sparsity = numpy.array([1, 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/(sparsity[i]*coeff)) for i in range(len(x))])
y = numpy.array([2.324241983,	4.435789726,	5.921728955,	6.994841305,	7.768480633,	7.877675456,	2.40759297,	4.586554241,	6.299510213,	8.12722686,	9.305433163,	10.46236322,	11.17621879,	12.11379701,	12.59296758,	12.89725147,	13.31887791,	13.67491446,	13.82070674,	14.1110796,	14.02404563,	14.08680855,	14.40473858,	14.85546377,	14.853169,	14.87415719])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
axs[1][0].plot(x[:6], y[:6], 'o', markersize=12, color='#d73027')
axs[1][0].plot(x[6:], y[6:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 22, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[1][0].plot(xx, yy1, '-', lw=5, color='#d73027')
axs[1][0].plot(xx, yy2, '-', lw=5, color='#4575b4')
axs[1][0].set_title(f"Mamba-CS", fontsize=25)
# axs[1][0].set_xlabel('Batch size')
# axs[1][0].set_ylabel('Throughput (queries/sec)')
axs[1][0].set_ylim([0, None])
# plt.savefig('mamba_cs.png')


rms = 0.0
for i in range(0, len(y)):
    if i < 6:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[1][0].text(-0.18, 14.7, f'RMSE={rms:.2f}', fontsize=25)

coeff=1.8

x = numpy.array([1, 1, 2, 3, 4])
sparsity = numpy.array([1, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/(sparsity[i]*coeff)) for i in range(len(x))])
y = numpy.array([0.2833587728, 0.3354076283, 0.6216146478,	0.8325194069, 1.005476933])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
axs[0][1].plot(x[:1], y[:1], 'o', markersize=12, color='#d73027')
axs[0][1].plot(x[1:], y[1:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 5, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[0][1].plot(xx, yy1, '-', lw=5, color='#d73027')
axs[0][1].plot(xx, yy2, '-', lw=5, color='#4575b4')
axs[0][1].set_title(f"Mixtral-MATH", fontsize=25)
# axs[0][1].set_xlabel('Batch size')
# axs[0][1].set_ylabel('Throughput (queries/sec)')
axs[0][1].set_ylim([0, None])
# plt.savefig('mixtral_math.png')

rms = 0.0
for i in range(0, len(y)):
    if i < 1:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[0][1].text(-0.03, 1.08, f'RMSE={rms:.2f}', fontsize=25)


coeff=2.5

x = numpy.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sparsity = numpy.array([1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/(sparsity[i]*coeff)) for i in range(len(x))])
y = numpy.array([2.218754555,	3.231858367,	5.321407704,	2.173860115,	4.56952556,	6.453015914,	7.790000288,	8.963689885,	9.823115165,	10.55141706,	11.09739966,	11.3871612,	11.6119473])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
axs[1][1].plot(x[:3], y[:3], 'o', markersize=12, color='#d73027')
axs[1][1].plot(x[3:], y[3:], 'o', markersize=12, color='#4575b4')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 12, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/(0.25*coeff)))
axs[1][1].plot(xx, yy1, '-', lw=5, color='#d73027')
axs[1][1].plot(xx, yy2, '-', lw=5, color='#4575b4')
axs[1][1].set_title(f"Mamba-MATH", fontsize=25)
# axs[1][1].set_xlabel('Batch size')
# axs[1][1].set_ylabel('Throughput (queries/sec)')
axs[1][1].set_ylim([0, None])


rms = 0.0
for i in range(0, len(y)):
    if i < 3:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i])))**2
    else:
        rms += (y[i] - (out[0][0]+out[0][1]*numpy.log(x[i]**(1/(0.25*coeff)))))**2
import math
rms = math.sqrt(rms / len(y))
axs[1][1].text(-0.125, 12, f'RMSE={rms:.2f}', fontsize=25)

for i in range(0, 2):
    for j in range(0, 2):
        for label in axs[i][j].get_yticklabels():
            label.set_fontproperties(font1)
        for label in axs[i][j].get_xticklabels():
            label.set_fontproperties(font1)


# axs[0][1].set_xlabel('Batch size')
# axs[0][1].set_ylabel('Throughput (queries/sec)')
plt.text(-18, 7.5, "Throughput (queries/sec)", fontsize=27, rotation=90)
plt.text(-3.3, -3, "Batch size", fontsize=25, rotation=0)


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 7., forward=True)
fig.savefig('./analytical_throughput/sweep_per.png',bbox_inches='tight')
fig.savefig('./analytical_throughput/sweep_per.pdf',bbox_inches='tight')


