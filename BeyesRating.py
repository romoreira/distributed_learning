from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

movies_arr = [
        ['A', 60, 40],
        ['B', 6, 1],
        ['C', 6, 4],
        ['D', 1, 2],
        ['E', 10, 20],
               ];

movies= [dict(zip(['name','up','down'], a)) for a in movies_arr]

x = np.linspace(0, 1, 1001)

plt.figure(dpi = 200)
axes = plt.gca(frameon=False)


# Prior information, movies tend to be more average and less extreme
a0, b0 = (11,11)

for i in range(len(movies)):
    a = movies[i]['up'] + a0
    b = movies[i]['down'] + a0
    b1 = beta.pdf(x, a, b)

    p = axes.plot(x, b1, linewidth=0.5)
    p[0].set_label(f"{movies[i]['name']}  {movies[i]['up']}:{movies[i]['down']}" )
    axes.fill_between(x, b1, alpha=0.4)

    rank = beta.ppf(0.05, a, b)
    axes.axvline(x=rank, ymin=0, ymax=1, color=p[0].get_color(), linewidth=1)

    print(f"{movies[i]['name']}  {movies[i]['up']}:{movies[i]['down']} rank: {rank:0.2f}")

axes.grid(b=None, which='major', axis='both')

if a0 > 5:
    axes.set_xlim(0.2, 0.8)

plt.legend()