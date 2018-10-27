from textblob.classifiers import NaiveBayesClassifier as NBC

with open('train.json', 'r') as fp:
	cl = NBC(fp, format="json")

with open('test.json', 'r') as fp:
	print(str(cl.accuracy(fp,format="json")*100)+' %')

with open('file.txt', 'r') as fp:
	for i in fp.readlines():
		print(i[:-4]+': '+cl.classify(i))
