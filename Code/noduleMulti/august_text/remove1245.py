import os
import re

path1 = 'testdata/'
path2 = '1245/'

testdata = os.listdir(path1)
data = os.listdir(path2)

print(len(testdata))
print(len(data))

test = []
for one in testdata:
	index  = re.findall("\d+", one)[0]
	test.append(index)

print(len(testdata))

for one in data:
	index = re.findall("\d+", one)[0]
	if index in test:
		os.remove(path2 + one)

print(len(data))
