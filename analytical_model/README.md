## Analytical model

You can reproduce the results for Figure 13, 14 and 15 in the paper by running:
```bash
python3 ./analytical_capacity/plot.py
python3 ./analytical_throughput/plot.py
python3 ./analytical_gpu/plot.py
```
You will find each figure in analytical_capacity, analytical_throughput and analytical_gpu directories.
If you want to obtain the coefficients C2 and C4 for your model, update the x (batch size), sparsity, and y (throughput) values in plot.py. The coefficient corresponds C2 to out[0][0], and C4 corresponds to out[0][1].

