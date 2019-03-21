import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

dat = pandas.read_csv('whdr_vs_relight_1_epoch_2.csv')

dat = dat[dat['Recon'] < .1]

plt.figure()
sns_plot = sns.lmplot(x='WHDR',y='Relight',hue='alg', data=dat, lowess=True)
sns_plot.savefig('whdr_vs_relight.pdf')

plt.figure()
sns_plot = sns.lineplot(x='WHDR',y='Relight',hue='alg', data=dat, sort=False)
sns_plot.figure.savefig('whdr_vs_relight_line.pdf')

plt.figure()
sns_plot = sns.lineplot(x='WHDR',y='Relight',hue='alg', data=dat)
sns_plot.figure.savefig('whdr_vs_relight_line_2.pdf')


plt.figure()
sns_plot = sns.lmplot(x='WHDR',y='Recon',hue='alg', data=dat, lowess=True)
sns_plot.savefig('whdr_vs_recon.pdf')

plt.figure()
sns_plot = sns.lineplot(x='WHDR',y='Recon',hue='alg', data=dat, sort=False)
sns_plot.figure.savefig('whdr_vs_recon_line.pdf')

plt.figure()
sns_plot = sns.lineplot(x='WHDR',y='Recon',hue='alg', data=dat)
sns_plot.figure.savefig('whdr_vs_recon_line_2.pdf')

plt.figure()
sns_plot = sns.lmplot(x='WHDR Weight', y='Relight', hue='alg', data=dat, lowess=True)
sns_plot.savefig('whdr_weight_vs_relight.pdf')

plt.figure()
sns_plot = sns.lmplot(x='WHDR Weight', y='Recon', hue='alg', data=dat, lowess=True)
sns_plot.savefig('whdr_weight_vs_recon.pdf')

plt.figure()
sns_plot = sns.lmplot(x='WHDR Weight', y='WHDR', hue='alg', data=dat, lowess=True)
sns_plot.savefig('whdr_weight_vs_whdr.pdf')

