import numpy, scipy
import matplotlib.pyplot as plt
from scipy import optimize
coeff=1.8
plt.clf()
x = numpy.array([1, 1, 2, 3])
sparsity = numpy.array([1, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/sparsity[i]/coeff) for i in range(len(x))])
y = numpy.array([0.3886286631, 0.4490951743,	0.8150684641,	1.1513719])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
plt.plot(x[:1], y[:1], 'go')
plt.plot(x[1:], y[1:], 'bo')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 5, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/0.25/coeff))
plt.plot(xx, yy1, 'g-')
plt.plot(xx, yy2, 'b-')
plt.title(f"Mixtral math, C2={out[0][1]:.3f}, C3={out[0][0]:.3f}")
plt.xlabel('Batch size')
plt.ylabel('Throughput (queries/sec)')
plt.ylim([0, None])
plt.savefig('mixtral_math_A100_40GB.png')


coeff=1.8
plt.clf()
x = numpy.array([1, 2, 3, 4, 5, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17])
sparsity = numpy.array([1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/sparsity[i]/coeff) for i in range(len(x))])
y = numpy.array([0.4153268396,	0.6734088813,	0.9212998066,	0.9212998066,	0.9531133065, 0.4736763151,	0.8774646886,	1.1201166,	1.708054828,	1.800397168,	2.031587942,	2.189323022,	2.282276004,	2.305001484,	2.567787669,	2.589650462,	2.678787645,	2.693921973,	2.739929458])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
plt.plot(x[:5], y[:5], 'go')
plt.plot(x[5:], y[5:], 'bo')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 20, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/0.25/coeff))
plt.plot(xx, yy1, 'g-')
plt.plot(xx, yy2, 'b-')
plt.title(f"Mixtral math, C2={out[0][1]:.3f}, C3={out[0][0]:.3f}")
plt.xlabel('Batch size')
plt.ylabel('Throughput (queries/sec)')
plt.ylim([0, None])
plt.savefig('mixtral_math_A100_80GB.png')


coeff=1.8
plt.clf()
x = numpy.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17])
sparsity = numpy.array([1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
x_adjust = numpy.array([x[i]**(1/sparsity[i]/coeff) for i in range(len(x))])
y = numpy.array([0.5468527884,	0.9480089158,	1.27383748,	1.501140867,	1.656801793, 0.6716115184,	1.277928978,	1.491977043,	1.79810839,	2.167855772,	2.634467155,	2.936952442,	3.152775956,	4.135249112,	4.311939502,	4.394074678,	4.666207512,	4.635430356,	4.89877178])
out = scipy.optimize.curve_fit(lambda t,a,b: a+b*numpy.log(t),  x_adjust,  y)
print(out[0])
print([out[0][0]+out[0][1]*numpy.log(x_adjust[i])-y[i] for i in range(len(x_adjust))])
print()
plt.plot(x[:5], y[:5], 'go')
plt.plot(x[5:], y[5:], 'bo')
# plot a + b log(x), where a and b are the coefficients of the fit
xx = numpy.linspace(0, 20, 100)
yy1 = out[0][0]+out[0][1]*numpy.log(xx)
yy2 = out[0][0]+out[0][1]*numpy.log(xx**(1/0.25/coeff))
plt.plot(xx, yy1, 'g-')
plt.plot(xx, yy2, 'b-')
plt.title(f"Mixtral math, C2={out[0][1]:.3f}, C3={out[0][0]:.3f}")
plt.xlabel('Batch size')
plt.ylabel('Throughput (queries/sec)')
plt.ylim([0, None])
plt.savefig('mixtral_math_H100.png')