import os
import numpy as np
from gensim.models import Word2Vec
from trainingdata2 import extract

#Training data
sentences=extract()
print("sentences:  ",len(sentences))

#train model
model1=Word2Vec(sentences,min_count=5,size=300,window=5)
print(model1)
model1.save('model1.bin')
model=Word2Vec.load('model1.bin')
words=list(model.wv.vocab)
print("length of words  ",len(words))
def doc2vec(directory,filename):
	temp={}
	content=open(os.path.join('./',directory,filename),"r")
	for i in content:
		j=i.strip().split()
		for k in j:
			try:
				temp[k]+=1
			except:
				temp[k]=1
	return temp

directory={'business':[],'entertainment':[]}#,'politics':[],'sport':[],'tech':[]}
y={}
dd=0
for d in directory:
	for filename in os.listdir("./"+d+"/"):
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
	y[d]=[dd]*len(directory[d])
	dd=dd+1

'''
filedirectory={'business':'business.txt','entertainment':'entertainment.txt','politics':'politics.txt','sport':'sport.txt','tech':'tech.txt'}

for d in filedirectory:
	with open(filedirectory[d],'w') as filehandle:
		for i in directory[d]:
			for j in i:
				filehandle.write(str(j))
				filehandle.write(' ')
			filehandle.write('\n')

'''

print("classification started:")
total_accuracy=0
from sklearn import svm

fold=2
for i in range(fold):

	trainx=[]
	testx=[]
	trainy=[]
	testy=[]

	for d in directory:
		chunk_size=len(directory[d])/fold

		#testx.extend([directory[d][i] for i in range(i*length_d,i*length_d + length_d)])
		#testy.extend([y[d][i] for i in range(i*length_d,i*length_d + length_d)])

		testx = testx + directory[d][ i*chunk_size : chunk_size*(1+i) ]
		testy = testy + y[d][ i*chunk_size : chunk_size*(1+i) ]

		#trainx.extend([directory[d][i] for i in range(0,i*chunk_size)])
		#trainx.extend([directory[d][i] for i in range(i*chunk_size + chunk_size,len(directory[d]))]) 

		#trainy.extend([y[d][i] for i in range(0,i*chunk_size)])
		#trainy.extend([y[d][i] for i in range(chunk_size*(i+1),len(directory[d]))])

		trainx = trainx + directory[d][0:i*chunk_size] + directory[d][chunk_size*(i+1):len(directory[d])]
		trainy = trainy + y[d][0:i*chunk_size] + directory[d][chunk_size*(i+1):len(directory[d])]

	
	clf=svm.SVC(gamma='auto')
	clf.fit(trainx,trainy)
	countp=0
	countn=0
	for i in range(len(testx)):
		print(i)
		if clf.predict([testx[i]])[0] == testy[i]:
			countp+=1
		else:
			countn+=1
	print("accuracy   ",countn,countp,countp/(countp+countn))

	total_accuracy+=countp/(countp+countn)
print(total_accuracy/15)
