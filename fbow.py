import os
import numpy as np
from gensim.models import Word2Vec
from trainingdata2 import extract

#Training data
sentences=extract()
print("sentences:  ",len(sentences))

dict_Sentences={}

for i in sentences:
	for j in i:
		try:
			dict_Sentences[j]+=1
		except:
			dict_Sentences[j]=1

sorted_dict_list=list(sorted(dict_Sentences.items(),key=lambda kv:(kv[1],kv[0])))[::-1]

#train model
model1=Word2Vec(sentences,min_count=,size=100)
print(model1)
model1.save('model_fbow.bin')
model=Word2Vec.load('model1.bin')
words=list(model.wv.vocab)
print("length of vocab",words)
words_2000=[]
count=0

for i in sorted_dictt_list:
	if count==2000:
		break
	words_2000.append(i[0])
	count+=1

words=words_2000
print("length of words  ",len(words))

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

from sklearn import svm
clf=svm.SVC(gamma='auto')
clf.fit(trainx,trainy)
countp=0
countn=0
for i in range(len(testx)):
	print("classifying the file ",i)
	if clf.predict([testx[i]])[0] == testy[i]:
		countp+=1
	else:
		countn+=1
print("accuracy   ",countp,countn,countp/(countp+countn))

from joblib import dump,load
dump(clf,'clf_fbow.joblib')

'''

print("classification started:")
import sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,max_iter=4000,solver='lbfgs',multi_class='multinomial')
#from sklearn import svm
overall_accuracy=0
fold=15
for i in range(fold):
	print("for ",i)
	trainx=[]
	testx=[]
	trainy=[]
	testy=[]
	count=0
	for d in directory:
		chunk_size=int(len(dict_files[d])/fold)

		#testx.extend([directory[d][i] for i in range(i*length_d,i*length_d + length_d)])
		#testy.extend([y[d][i] for i in range(i*length_d,i*length_d + length_d)])

		testx = testx + dict_files[d][ i*chunk_size : chunk_size*(1+i) ]
		testy.extend([count for x in range(i*chunk_size,i*chunk_size + chunk_size)])

		#trainx.extend([directory[d][i] for i in range(0,i*chunk_size)])
		#trainx.extend([directory[d][i] for i in range(i*chunk_size + chunk_size,len(directory[d]))]) 

		#trainy.extend([y[d][i] for i in range(0,i*chunk_size)])
		#trainy.extend([y[d][i] for i in range(chunk_size*(i+1),len(directory[d]))])

		trainx = trainx + dict_files[d][0:i*chunk_size] + dict_files[d][chunk_size*(i+1):len(dict_files[d])]
		trainy.extend([count for x in range(0,i*chunk_size)])		
		trainy.extend([count for x in range(chunk_size*(i+1),len(dict_files[d]))])
		count+=1

	'''
	clf=svm.SVC(gamma="auto").fit(trainx,trainy)
	count_p=0
	count_n=0

	for j in range(len(testx)):
		print("predicted as class = ",int(clf.predict([testx[j]])),"    Actual class = ",int(testy[j]))
		if int(clf.predict([testx[j]]))==int(testy[j]):
			count_p+=1
		else:
			count_n+=1
	'''

	clf.fit(trainx,trainy)
	count_p=0
	count_n=0

	for i in range(len(testx)):
		print("predicted as class = ",int(clf.predict([testx[i]])),"    Actual class = ",int(testy[i]))
		if int(clf.predict([testx[i]]))==int(testy[i]):
			count_p+=1
		else:
			count_n+=1
	accuracy=(count_p*100)/(count_p+count_n)
	print("correctly predicted ",count_p,"wrong predictions",count_n,"the accuracy is = ",accuracy)
	overall_accuracy+=accuracy

print("the overall accuracy is",overall_accuracy/fold)