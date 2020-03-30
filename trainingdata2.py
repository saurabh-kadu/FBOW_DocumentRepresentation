import os
from nltk.tokenize import word_tokenize
import re
from stop_words import get_stop_words
def extract():
	data=[]
	stop_words = get_stop_words('english')
	directory=['business','entertainment','politics','sport','tech']
	for i in directory:
		for filename in os.listdir("./"+i+"/"):
			temp=[]
			content=open(os.path.join('./',i,filename),"r")
			for j in content:
				#temp.extend(j.rstrip('\n').split())
				t=word_tokenize(j.rstrip('\n'))
				temp.extend(t)
			temp=[w.lower() for w in temp if w not in stop_words]
			temp=[w for w in temp if w not in stop_words]
			temp=[w for w in temp if w not in ['.',',','!','@','#','$','%','^','&','*','(',')','[',']','{','}','?','-','_','+','=','``','`','~','/',';',':','|']]
			data.append(temp)
		#print(data)
	#print(len(data))
	return data
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
