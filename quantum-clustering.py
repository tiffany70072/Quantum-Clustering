import matplotlib.pyplot as plt
import numpy as np
import pickle

from math import exp
from numpy import array
from numpy import empty
from numpy import genfromtxt
from numpy import zeros
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Model(object):
	def __init__(self):
		self.normal_const = []
		self.phi = []
		self.v = []
		self.numK = 8
		self.beta = 0.8
		
	def load_data(self):
		raw_data = array(genfromtxt('data.csv', delimiter=','))
		self.id = raw_data[:, 1]
		self.x = raw_data[:, 2:]

	def adjust_data(self):
		self.x = np.delete(self.x, 5, 1)
		self.x = np.delete(self.x, 3, 1)
		self.x = np.delete(self.x, 69, 0)
		self.m = self.x.shape[1]
		self.n = self.x.shape[0]
		print "n, m = ", self.x.shape

	def compute_delta(self):
		self.delta = (4.0 / (self.m + 2.0) / self.n) ** (1 / (self.m + 4.0))

	def compute_normal_const(self):
		self.normal_const = []
		for col in range(self.m): 
			total = 0.0
			for i in range(self.n): 
				total += self.x[i][col]
			total /= float(self.n)
			self.normal_const.append(total)

	def normalize(self):
		for col in range(self.m): 
			self.x[:, col] /= self.normal_const[col]

	def compute_phi(self, x0):
		phi = 0
		for i in range(self.n): 
			phi += exp(-(euclidean(x0, self.x[i]) ** 2) / 2 / self.delta / self.delta)
		return phi

	def same(self, xi, xj):
		if xi == xj: 
			return 1
		else: 
			return 0

	def compute_d(self): 
		# Can only compute half range.
		self.d = []
		for i in range(self.n):
			self.d.append([])
			for j in range(self.n):
				self.d[i].append(0)
				for k in range(self.m): 
					self.d[i][j] += self.same(self.x[i][k], self.x[j][k])
				self.d[i][j] = 1 - self.d[i][j] / float(self.m)
		self.d = np.array(self.d)

	def compute_performance(self):
		co = 0
		score = 0
		pscore = 0
		rscore = 0
		nscore = 0
		for i in range(self.n-1):
			for j in np.arange(i+1, self.n):
				result = self.same(self.c[i], self.c[j])
				answer = self.same(self.id[i], self.id[j])
				if result == 1 and answer == 1: 
					score += 1
				elif result == 1 and answer == 0: 
					pscore += 1
				elif result == 0 and answer == 1: 
					rscore += 1
				else: 
					nscore += 1

		# print "Performance: ", 
		# p = score / float(pscore + score)
		# r = score / float(rscore + score)
		# b = 0.5
		# acc1 = (1 + b ** 2) * p * r / float((b ** 2) * p + r)
		# acc2 = 2 * p * r / float(p + r)
		ri = (score + nscore) / float(score + pscore + rscore + nscore)
		print "RI = %.3f" % ri
		# print "total = %.3f" % (2 * ri * acc2 / (ri + acc2)), score + nscore
		# print score, pscore, rscore, nscore, score + nscore

	def pca(self, dim):
		print "Dimension Reduction"
		pca = PCA(n_components=dim, copy=False) 
		self.x = pca.fit_transform(self.x)
		print pca.explained_variance_ratio_
		self.m = dim

	def pcaVisualization(self):
		colors_list = []
		colors = ['gold', 'orangered', 'b', 'c', 'salmon', 'aqua', 
		'gold', 'greenyellow', 'mistyrose', 'coral']
		print "Visualization shape = ", self.x.shape

		for i in range(self.id.shape[0]): 
			colors_list.append(colors[int(self.id[i])-1])
		plt.scatter(self.x[:, 0], self.x[:, 1], c=colors_list, marker='o', s=40)
		plt.show()

	def kmeans(self, num_c):
		print "KMeans Clustering = ", num_c
		temp = self.x
		self.kmeans = KMeans(n_clusters=num_c, max_iter=500).fit(temp)
		# print "Center = ", self.kmeans.cluster_centers_

		self.c = []
		for i in range(self.x.shape[0]): 
			self.c.append(self.kmeans.predict([self.x[i]]))

	def compute_potential(self):
		self.v = zeros(self.n, float)
		for i in range(self.n):  # Point to get potential energy.
			sigma = 0
			for k in range(self.n):  # Charge point.
				coefficient = -(euclidean(self.x[k], self.x[i]) ** 2) / 2.0 / self.delta / self.delta
				sigma += (euclidean(self.x[k], self.x[i]) ** 2) * exp(coefficient)
			phi = self.compute_phi(self.x[i])
			self.v[i] = 1 / 2.0 / self.delta / self.delta / phi * sigma

	def potential_visualization(self):
		print np.amax(self.v)
		v_max = np.amax(self.v)
		self.v = self.v / v_max
		c_list = []
		for i in range(self.n):
			if self.v[i] < 0.2: 
				c_list.append('greenyellow')
			elif self.v[i] < 0.4: 
				c_list.append('yellow')
			elif self.v[i] < 0.55: 
				c_list.append('gold')
			elif self.v[i] < 0.7: 
				c_list.append('orange')
			elif self.v[i] < 0.85: 
				c_list.append('r')
			else: 
				c_list.append('b')

		print len(c_list)
		plt.scatter(self.x[:, 0], self.x[:, 1], c=c_list, marker='o', alpha=0.5, s=75)  # alpha = 0.8
		plt.show()

	def build_space(self, num):
		y = zeros((num * num, 2), float)
		for i in range(num):
			for j in range(num):
				y[i * num + j][0] = float(i)
				y[i * num + j][1] = float(j)
		
		y = y * 2 - 8
		v = zeros(num * num, float)
		for i in range(num):  # Point to get potential energy.
			for j in range(num):
				sigma = 0
				for k in range(self.n): # Charge point.
					coefficient = -(euclidean(self.x[k], y[i * num + j]) ** 2) / 2.0 / self.delta / self.delta
					sigma += (euclidean(self.x[k], y[i * num + j]) ** 2) * exp(coefficient)
				phi = self.compute_phi(y[i * num + j])
				v[i * num + j] = 1 / 2.0 / self.delta / self.delta / phi * sigma

		y = np.concatenate((y, self.x), axis=0)
		num = num * num + self.n
		v = np.concatenate((v, self.v), axis=0)
		v_max = np.amax(v)
		v_max = 10
		v = v / v_max * 5
		print "here", v_max, y.shape, v.shape, num

		c_list = []
		for i in range(num):
			if v[i] < 0.2: 
				c_list.append('greenyellow')
			elif v[i] < 0.4: 
				c_list.append('yellow')
			elif v[i] < 0.55: 
				c_list.append('gold')
			elif v[i] < 0.7: 
				c_list.append('orange')
			elif v[i] < 0.85: 
				c_list.append('r')
			else: 
				c_list.append('b')

		print len(c_list)
		plt.scatter(y[:, 0], y[:, 1], c=c_list, marker='o', alpha=0.5, s = 75)  # alpha = 0.8
		plt.show()

	def findIndex(self, goal, listt):
		for index, item in enumerate(listt):
			if item == goal: 
				break
			else: 
				index = -1
		return index

	def normalize2(self, arr):
		arr_max = np.amax(arr)
		arr = arr / arr_max
		return arr

	def addToCluster(self, cl):
		self.c = []
		for i in range(self.x.shape[0]): 
			self.c.append(0)
		for i in range(len(cl)):
			for j in range(len(cl[i])):
				self.c[cl[i][j]] = i

	def qc(self):
		x = self.x
		v = self.v
		v = self.normalize2(v)
		time = 0
		c = []  # 2 dim, 1 for cluster index, 1 for members in it.
		
		while x.shape[0] != 0:
			time += 1
			v_min = np.amin(v)
			c.append([])
			index = self.findIndex(v_min, v)
			
			for i in range(x.shape[0]):
				if self.d[index][i] <= self.beta: 
					c[time - 1].append(i)
			length = len(c[time - 1])
			for i in range(length):
				remove = c[time - 1][length - i - 1]
				x = np.delete(x, remove, 0)
				v = np.delete(v, remove)

		print "Cluster = ", len(c)
		self.addToCluster(c)

	def visual(self):
		colors_list = []
		colors = ['gold', 'b', 'orangered', 'aqua', 'c', 'salmon', 
		'#30a2da', 'g', 'mistyrose', 'coral', 'pink', 'g', 'r', 
		'#C44E52', '#6d904f', '#FFC400', 'm', 'g', 'k']  
		print "Visualization shape = ", self.x.shape
		
		for i in range(self.x.shape[0]): 
			colors_list.append(colors[self.c[i]])
		plt.scatter(self.x[:, 0], self.x[:, 1], c=colors_list, marker='o', s=40)
		plt.show()


model = Model()
model.load_data()
model.adjust_data()
model.compute_normal_const()
model.normalize()
# model.pca(2)
# model.pcaVisualization()
# exit()

# kmeans clustering algorithm.
# model.kmeans(model.numK)
# model.compute_performance()
# model.visual()
# exit()

# Quantum clustering algorithm.
model.compute_delta()
model.compute_potential()
# model.potential_visualization()
model.compute_d()
li = [0.8, 0, 0.2, 0.4, 0.6, 0.8, 1]
for i in range(1):
	model.beta = li[i]
	model.qc()
	model.compute_performance()
	model.pca(2)
	model.visual()
print model.x.shape

# model.build_space(10)
