import os
import numpy as np
from gensim.models import Word2Vec
from trainingdata2 import extract,doc2vec

#Training data
sentences=extract()
print("sentences:  ",len(sentences))

#train model
model_fbow_1=Word2Vec(sentences,min_count=3,size=6000,window=8)
print(model_fbow_1)
model_fbow_1.save('model_fbow_1.bin')
model=Word2Vec.load('model_fbow_1.bin')
words=list(model.wv.vocab)
print("length of vocab",len(words))

directory={'business':[],'entertainment':[],'politics':[],'sport':[],'tech':[]}
for d in directory:
	for filename in os.listdir("./"+d+"/"):
		print("directory  and filename  " ,d,filename)
		file=doc2vec(d,filename)
		filevector=[[] for i in range(len(file))]
		for w in words:
			i=0
			for fw in file:
				try:
					simi=model.wv.similarity(w,fw)
					if simi>0:
						filevector[i].append(model.wv.similarity(w,fw))
					else:
						filevector[i].append(0)
				except:
					filevector[i].append(0)
				i+=1
		temp=0
		temparray=[]
		for j in range(len(words)):
			i=0
			for ff in file:
				temp=temp+file[ff]*filevector[i][j]
				i=i+1
			temparray.append(temp)
		directory[d].append(temparray)

filedirectory={'business':'business_full.txt','entertainment':'entertainment_full.txt','politics':'politics_full.txt','sport':'sport_full.txt','tech':'tech_full.txt'}
for d in filedirectory:
	with open(filedirectory[d],'w') as filehandle:
		for i in directory[d]:
			for j in i:
				filehandle.write(str(j))
				filehandle.write(' ')
			filehandle.write('\n')


print("classification started:")
trainx=[]
testx=[]
trainy=[]
testy=[]
count=0
for d in directory:
	trainx.extend([directory[d][i] for i in range(int(len(directory[d])*0.8))])
	trainy.extend([count for i in range(int(len(directory[d])*0.8))])
	testx.extend([directory[d][i] for i in range(int(len(directory[d])*0.8),int(len(directory[d])))])	
	testy.extend([count for i in range(int(len(directory[d])*0.8),int(len(directory[d])))])
	count+=1



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

from joblib import dump,load
dump(clf,'clf_fbow_full.joblib')
