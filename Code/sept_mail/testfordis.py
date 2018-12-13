import csvTools
labels = csvTools.readCSV('pixdis.csv')

print(len(labels[0]))
labels = labels[0]

max = float(labels[0])
min = float(labels[0])

for one in labels:
    temp = float(one)
    if temp > max:
        max = temp
    elif temp < min:
        min = temp
print(max, min)