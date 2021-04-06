from __future__ import division
import os
import sys
import time
import numpy as np
from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
from multiprocessing import Pool
import string
import warnings
warnings.filterwarnings("ignore")

mpl.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

################## Fourier #######################

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]

def random_fourier(seed):
	np.random.seed(seed)
	Coeffs = np.random.rand(2,fmax)
	y = np.multiply(Template,Coeffs)
	y = np.sum(y,axis=(1,2))
	l,h=np.sort(np.random.rand(2))
	y = MinMaxScaler(feature_range=(l,h)).fit_transform(y.reshape(-1, 1)).reshape(-1)
	# y = MinMaxScaler(feature_range=(l,h)).fit_transform(y)
	return y

################## Lines #######################
def line_family(seed):
	np.random.seed(seed)
	y1 = np.random.random()
	y2 = np.random.random()
	y = np.linspace(y1,y2,1024)
	return y

################## Cosines #######################
def cos_family(seed):
	np.random.seed(seed)
	l,h=np.sort(np.random.rand(2))
	A = 0.5*(h-l)
	shift = 0.5*(h+l)
	f = 20*np.random.random()
	theta = 2*pi*np.random.random()
	y=A*np.cos(2*pi*f*x + theta)+shift
	return y

############### Polynomial Fit #####################
def random_poly_fit(seed):
	np.random.seed(seed)
	l=0
	h=1
	degree = np.random.randint(2,11)
	c_points = np.random.randint(2,32)
	cx = np.linspace(0,1,c_points)
	cy = np.random.rand(c_points)
	z = np.polyfit(cx, cy, degree)
	f = np.poly1d(z)
	y = f(x)
	if degree==1: 
		l,h=np.sort(np.random.rand(2))
	y = MinMaxScaler(feature_range=(l,h)).fit_transform(y.reshape(-1, 1)).reshape(-1)
	return y

############### B Splines Fit #####################
def random_bspline(seed):
	np.random.seed(seed)
	l=0
	h=1
	degree = 3
	c_points = np.random.randint(4,32)
	cx = np.linspace(0,1,c_points)
	cy = np.random.rand(c_points)
	z = interpolate.splrep(cx, cy, k=degree)
	y = interpolate.splev(x, z)
	# l,h=np.sort(np.random.rand(2))
	y = MinMaxScaler(feature_range=(l,h)).fit_transform(y.reshape(-1, 1)).reshape(-1)
	return y

########### Cubic Splines Interpolation #############
def random_cubic_spline(seed):
	np.random.seed(seed)
	l=0
	h=1
	c_points = np.random.randint(4,32)
	cx = np.linspace(0,1,c_points)
	cy = np.random.rand(c_points)
	z = interpolate.CubicSpline(cx, cy)
	y = z(x)
	# l,h=np.sort(np.random.rand(2))
	y = MinMaxScaler(feature_range=(l,h)).fit_transform(y.reshape(-1, 1)).reshape(-1)
	return y

# func_families = [line_family, cos_family,random_fourier]

func_families = [random_poly_fit, 
				random_bspline,
				random_cubic_spline]

markers = ['.',',','o','v','^','<','>',
			'1','2','3','4','s','p','*',
			'h','H','+','x','D','d','|','_','']

linestyles = ['-','--','-.',':','']

colors = ['b','g','r','c','m','y','k']
locations = ['center', 'left', 'right']
xlocations = ['center', 'left', 'right']
ylocations = ['center', 'bottom', 'top']
rotations = [0,90,180,270]
alphabet = list(string.ascii_letters + string.digits + '!"#%&\'()*+,-.:;<=>?@[]^_`{|}~' + ' ')

sty = style.available

N = 10**3 			# Size of the dataset, i.e, number of images to be generated 	
K = 5				# Maximum number of plots in a single image		
# chunk_size = 100
my_dpi = 96			


# ax = plt.axes([0,0,1,1], frameon=False)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax.set_ylim(0,1)
# ax.set_xlim(0,1)

DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR): 
	os.makedirs(DATA_DIR)
	os.makedirs(os.path.join(DATA_DIR,'train'))
	os.makedirs(os.path.join(DATA_DIR,'test'))

x = np.linspace(0,1,1024)

########## Templates for Fourier ################
# fmax = 20
# Template = np.zeros([1024,2,fmax])
# for f in range(fmax):
# 	Template[:,0,f] = np.cos(2*pi*(f+1)*x)
# 	Template[:,1,f] = np.sin(2*pi*(f+1)*x)
################################################

def generate_plot(inp):
	i,seed = inp
	seed=seed
	np.random.seed(seed)
	k = np.random.randint(1,K+1)
	Y = []

	aspect_ratios = [1.,3./2.,2./3.,4./3.,3./4.,16./9.,9./16.]

	plt.figure(figsize=(1024/my_dpi, 1024*np.random.choice(aspect_ratios)/my_dpi), dpi=my_dpi)
	mpl.rcParams['savefig.pad_inches'] = 0
	plt.margins(x=np.clip(np.abs(np.random.normal(0,0.1)),0,1),y=np.clip(np.abs(np.random.normal(0,0.1)),0,1))


	for idx in range(k):
	# Choose parameters randomly
		func = np.random.choice(func_families)
		marker = np.random.choice(markers)
		ls = np.random.choice(linestyles)
		c = np.random.choice(colors)
		mfc = np.random.choice(colors)
		lw = 5*np.random.random()+2
		ms = 5*np.random.random()+2

		if np.random.uniform()<0.1: func = line_family  

		label = ''.join(np.random.choice(alphabet, size=np.random.randint(1,15)))

		y = func(seed*(N+idx)%(2**31))
		Y.append(y)

		plt.grid(np.random.choice([True,False]))

		style.use(np.random.choice(sty))

		# Avoid boundary conditions. This is done to avoid empty plots.
		bndry = False
		if marker=='' and ls=='':
			bndry = True

		if bndry:
			# myplot = plt.plot(x,y,c=c)
			plt.plot(x,y,c=c,label=label)
		else:
			# myplot = plt.plot(x,y,c=c,ls=ls,lw=lw, marker=marker,ms=ms,mfc=mfc)
			plt.plot(x,y,c=c,ls=ls,lw=lw, marker=marker,ms=ms,mfc=mfc,label=label)

	if (i/N)<0.8:
		phase = 'train'
	else:
		phase = 'test'

	plt.title(label=''.join(np.random.choice(alphabet, size=np.random.randint(1,30))),fontsize=np.random.randint(20,50),loc=np.random.choice(locations))
	plt.xlabel(''.join(np.random.choice(alphabet, size=np.random.randint(1,20))), fontsize=np.random.randint(10,30), loc=np.random.choice(xlocations))
	plt.ylabel(''.join(np.random.choice(alphabet, size=np.random.randint(1,20))), fontsize=np.random.randint(10,30), loc=np.random.choice(ylocations))
	plt.xticks(fontsize=np.random.randint(10,45), rotation=np.random.choice(rotations))
	plt.yticks(fontsize=np.random.randint(10,45), rotation=np.random.choice(rotations))
	plt.legend(loc=0)

	plt.savefig(os.path.join(DATA_DIR,phase,'%06d.jpg'%i),dpi=my_dpi)
	np.save(os.path.join(DATA_DIR,phase,'%06d.npy'%i),np.array(Y))
	plt.clf()
	plt.close('all')

if __name__ == '__main__':
	t = time.time()
	# chunk_list = list(chunks(range(N), chunk_size))
	with Pool(int(mp.cpu_count())//2) as p:

		# np.random.seed(45)
		# seeds = np.random.randint(2**30, N)

		p.map(generate_plot, zip(range(N),range(N)))
		
		# for i, _ in enumerate(p.imap_unordered(generate_plot, range(N)), 1):
		# 	sys.stderr.write('\rProgress: {0:%}'.format(i/N))

		# for i, chunk in enumerate(chunk_list,1):
		# 	p.map(generate_plot, chunk)
		# 	sys.stderr.write('\rProgress: {0:%}'.format(i/(len(chunk_list))))

	print("\n Total time taken: %f"%(time.time()-t))