#######################Direselign Addis 106999405 ###########################################
################### This is the implmentation of PCA for Machine learning Homework Two########
########## We can run this from linux environment or from windows using Anaconda#############
#############################################################################################
import pandas as pd
import numpy as np
from pylab import *
from matplotlib import pyplot as plt
import math
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
url = 'iris.csv'
list1=[]
dframe=pd.read_csv(url,dtype={'sepal_length': np.float64,'sepal_width': np.float64,'petal_length': np.float64,'petal_width': np.float64,'class': np.object})
df = pd.DataFrame(dframe)
x1 = StandardScaler().fit_transform(df.iloc[0:len(df), 0:4])
cov=np.cov(x1.T)
print("covariance:\n",cov)
evalues, evectors = np.linalg.eig(cov)
print("eigen vectors:\n",evalues)
print("eigen values:\n",evectors)
eig_pairs = [(np.abs(evalues[i]), evectors[:,i]) for i in range(len(evalues))]
eig_pairs.sort()
eig_pairs.reverse()
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n',matrix_w)
y= x1.dot(matrix_w)
newdata=pd.DataFrame(y,columns=["PC1","PC2"])
newdata['class']=df['class']
print(newdata)
sumlist=[]
sumlist2=[]
variance_sum=0
accuracy=0
for indexer in range(10):
	train, test = train_test_split(newdata, test_size=0.3)
	def different_kvalues(k):
		result2=pd.DataFrame()
		correct_counter=0
		updated_train=train.iloc[0:len(train), 0:2]
		updated_test=test.iloc[0:len(test), 0:2]
		for x in range(len(updated_test)):
			result=pd.DataFrame()
			for y in range(len(updated_train)):
				sum1=(updated_test.iloc[x]-updated_train.iloc[y])
				power1= np.power(sum1,2) 
				df = pd.DataFrame(power1)
				result0=df.T
				result0['sum']=result0['PC1']+result0['PC2']
				Edistance=result0.apply(np.sqrt)
				Edistance['class']=train.iloc[y,2]
				result=result.append(Edistance)
			result.reset_index(drop=True,inplace=True)
			result['rank'] = result['sum'].rank(ascending=1,method='max')
			rank1= result['rank'] <=k
			result1=result[rank1]
			result1['freq'] = result1.groupby(['class']).size()
			sorted=result1.sort_values(by=['freq'],ascending=[False])
			result2=result2.append(sorted.head(1))
		for t in range (len(test)):
			if test.iloc[t][2]==result2.iloc[t][3]:
				correct_counter=correct_counter+1
		accuracy=correct_counter/len(test)
		print('Accuracy at testing {}: ={:0.3f}={:0.3f}'.format(indexer+1,accuracy,accuracy*100),'%')
		sumlist.append(accuracy)
		sumlist2.append(accuracy*100)
	different_kvalues(7)
sum=sumlist[0]+sumlist[1]+sumlist[2]+sumlist[3]+sumlist[4]+sumlist[5]+sumlist[6]+sumlist[7]+sumlist[8]+sumlist[9]
ave=(sum/10)
for var in range(10):
	variance_sum=variance_sum+pow((sumlist[var]-ave),2);
print("\nAverage Accuracy={:0.1f}".format(ave*100))
print("\n")
variance_pc=np.var(sumlist2)
print('Accuracy Variance: {:0.3f}'.format(variance_sum/10))
print('Accuracy Variance in %: {:0.13}'.format(variance_pc),'%')
for list in range(10):
	list1.append(list+1)
plt.title("Testing phases versus accuracy graph")
plt.xlabel("Test")
plt.ylabel("accuracy")
plt.plot(list1,sumlist2,'go--', linewidth=2, markersize=12)
plt.show()

  

