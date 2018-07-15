#######################Direselign Addis 106999405 ###########################################
################### This is the implmentation of LDA for Machine learning Homework Two########
########## We can run this from linux environment or from windows using Anaconda#############
#############################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
url = 'iris.csv'
dframe=pd.read_csv(url,dtype={'sepal_length': np.float64,'sepal_width': np.float64,'petal_length': np.float64,'petal_width': np.float64,'class': np.object})
df = pd.DataFrame(dframe)
list1=[]
list2=[]
setosa = pd.DataFrame(columns=['sepal_length','sepal_width','petal_length','petal_width','class'])
versicolor = pd.DataFrame(columns=['sepal_length','sepal_width','petal_length','petal_width','class'])
virginica = pd.DataFrame(columns=['sepal_length','sepal_width','petal_length','petal_width','class'])
for t in range (len(df)):
		if df.iloc[t][4]=='Iris-setosa':
			setosa=setosa.append(df.iloc[t])
		if df.iloc[t][4]=='Iris-versicolor':
			versicolor=versicolor.append(df.iloc[t])
		if df.iloc[t][4]=='Iris-virginica':
			virginica=virginica.append(df.iloc[t])
print(setosa)
m1=np.mean(setosa)
m2=np.mean(versicolor)
m3=np.mean(virginica)
print(setosa)
print(versicolor)
print(np.mean(versicolor))
print(virginica)
print(np.mean(virginica))
smean=(np.append(np.mean(setosa['sepal_length']),np.append(np.mean(setosa['sepal_width']),(np.append(np.mean(setosa['petal_length']),np.mean(setosa['petal_width']))))))
cmean=(np.append(np.mean(versicolor['sepal_length']),np.append(np.mean(versicolor['sepal_width']),(np.append(np.mean(versicolor['petal_length']),np.mean(versicolor['petal_width']))))))
vmean=(np.append(np.mean(virginica['sepal_length']),np.append(np.mean(virginica['sepal_width']),(np.append(np.mean(virginica['petal_length']),np.mean(virginica['petal_width']))))))
appendrow1=np.vstack((smean,cmean))
appendrow2=np.vstack((appendrow1,vmean))
print(appendrow2)
overall=(np.mean(appendrow2,axis=0))
x1 =(setosa.iloc[0:len(setosa), 0:4])
x2 = (versicolor.iloc[0:len(versicolor), 0:4])
x3 = (virginica.iloc[0:len(virginica), 0:4])
x11=(x1-m1).T.dot((x1 - m1))
x22=(x2-m2).T.dot((x2 - m2))
x33=(x3-m3).T.dot((x3 - m3))
S1=x11+x22+x33
convert1=pd.DataFrame(smean)
convert2=pd.DataFrame(cmean)
convert3=pd.DataFrame(vmean)
mean2=(cmean-overall).T.dot((cmean - overall))
mean3=(vmean-overall).T.dot((vmean - overall))
convert11=((50*(convert1.T)-overall).T.dot((convert1.T)-overall))
convert22=((50*(convert2.T)-overall).T.dot((convert2.T)-overall))
convert33=((50*(convert3.T)-overall).T.dot((convert3.T)-overall))
S2=convert11+convert22+convert33
evalues, evectors = np.linalg.eig(np.linalg.inv(S1).dot(S2))
print(evectors)
print(evalues)
eig_pairs = [(np.abs(evalues[i]), evectors[:,i]) for i in range(len(evalues))]
eig_pairs.sort()
eig_pairs.reverse()
print('Eigenvalues in descending order:')
for i in eig_pairs:
	print(i[0])
W=np.hstack((eig_pairs[0][1].reshape(4,1), 
	eig_pairs[1][1].reshape(4,1)))
print(W)
x1 =(df.iloc[0:len(df), 0:4])
lda = x1.dot(W)
lda['class']=df['class']
lda.columns = ['LDA1', 'LDA2','class']
print(lda)
sumlist=[]
sumlist2=[]
variance_sum=0
accuracy=0
for indexer in range(10):
	train, test = train_test_split(lda, test_size=0.3)
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
				result0['sum']=result0['LDA1']+result0['LDA2']
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
		print('Accuracy at testing {}: ={:0.3f}={:0.1f}'.format(indexer+1,accuracy,accuracy*100),'%')
		sumlist.append(accuracy)
		sumlist2.append(accuracy*100)
	different_kvalues(7)
sum=sumlist[0]+sumlist[1]+sumlist[2]+sumlist[3]+sumlist[4]+sumlist[5]+sumlist[6]+sumlist[7]+sumlist[8]+sumlist[9]
ave=(sum/10)
for var in range(10):
	variance_sum=variance_sum+pow((sumlist[var]-ave),2);
print("\nAverage Accuracy={:0.3f}".format(ave*100))
print("\n")
variance_pc=np.var(sumlist2)
print('Accuracy Variance: {:0.3f}'.format(variance_sum/10))
print('Accuracy Variance in %: {:0.3F}'.format(variance_pc),'%')
for list in range(10):
	list1.append(list+1)
plt.title("Testing phases versus accuracy graph")
plt.xlabel("Test")
plt.ylabel("accuracy")
plt.plot(list1,sumlist2, 'go--', linewidth=2, markersize=12)
plt.show()
