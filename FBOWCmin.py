
import os
import numpy as np
from gensim.models import Word2Vec
from trainingdata2 import extract,doc2vec

#Training data
sentences=extract()
print("sentences:  ",len(sentences))

#Train model
modelMin=Word2Vec(sentences,min_count=3)
print(modelMin)
modelMin.save('modelMin.bin')
model=Word2Vec.load('modelMin.bin')
words=list(model.wv.vocab)
print("words : ",len(words))

#Clustering
from sklearn.cluster import KMeans
import numpy as np
X=[]
for i in range(len(words)):
	X.append(list(model.wv[words[i]]))
#Xinp=np.array(X)
print("Clustering :: : : :")
kmeans=KMeans(n_clusters=2000,random_state=None).fit(X)

labels=list(kmeans.labels_)
print("Done Clustering")

Cluster={}
ClusterWord={}
for i in range(2000):
	Cluster[i]=[]
	ClusterWord[i]=[]

for i in range(len(labels)):
	Cluster[labels[i]].append(X[i])
	ClusterWord[labels[i]].append(words[i])

directory={'business':[],'entertainment':[],'politics':[],'sport':[],'tech':[]}
y={}
dd=0
for d in directory:
	for filename in os.listdir("./"+d+"/"):
		print("filename : ",filename,d)
		file=doc2vec(d,filename)
		filevector=[[] for i in range(len(file))]
		for c in ClusterWord:	#for each Cluster
			i=0
			for fileword in file: #for each word in afile
				minm=float('Inf')
				for word in ClusterWord[c]:   #for each word of a cluster
					try:
						temp=model.wv.similarity(word,fileword)
					except:
						temp=0
					if temp<minm and temp>0:
						minm=temp
					elif temp<minm and temp<=0:
						minm=0
				filevector[i].append(minm)
				i=i+1

		temp=0
		temparray=[]
		for j in range(len(ClusterWord)): #Multiplying with frequency
			i=0
			for f in file:
				temp=temp+file[f]*filevector[i][j]
				i=i+1
			temparray.append(temp)
		directory[d].append(temparray)
	y[d]=[dd]*len(directory[d])
	dd=dd+1



trainx=[]
testx=[]
trainy=[]
testy=[]
for d in directory:
	trainx.extend([directory[d][i] for i in range(int(len(directory[d])*0.8))])
	trainy.extend([y[d][i] for i in range(int(len(directory[d])*0.8))])
	testx.extend([directory[d][i] for i in range(int(len(directory[d])*0.8),int(len(directory[d])))])	
	testy.extend([y[d][i] for i in range(int(len(directory[d])*0.8),int(len(directory[d])))])


import sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,max_iter=4000,solver='lbfgs',multi_class='multinomial').fit(trainx,trainy)

count_p=0
count_n=0
for i in range(len(testx)):
	print(int(clf.predict([testx[i]])),int(testy[i]))
	if int(clf.predict([testx[i]]))==int(testy[i]):
		count_p+=1
	else:
		count_n+=1
print(count_p,count_n,(count_p*100)/(count_p+count_n))
print(clf.score(testx,testy)*100)

'''
from sklearn.linearmodel import 
clf=svm.SVC(gamma='auto')
clf.fit(trainx,trainy)
countp=0
countn=0
for i in range(len(testx)):
	if clf.predict([testx[i]])[0] == testy[i]:
		countp+=1
	else:
		countn+=1
print("accuracy   ",countn,countp,countp/(countp+countn))
'''
from joblib import dump,load
dump(clf,'FBOWmin.joblib')
