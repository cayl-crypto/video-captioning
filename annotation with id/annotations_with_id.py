
import os
import numpy as np

import csv


dates = []
scores = []

with open('C:\\Users\\pc\\PycharmProjects\\video-captioning\\annotations\\Kitap5.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        dates.append(row[0])
        scores.append(row[1])

print(dates)
print(scores)


fileName = 'vid_id.txt'

fileObj = open(fileName, "r") #opens the file in read mode
vid= fileObj.read().splitlines()

#file = open('id.txt', "w+") #opens the file in read mode
id=[]
for j in range(len(dates)):
   for i in range(len(vid)):
       if (vid[i]== scores[j]):
          id.append(dates[j])

   else:
       continue


#print(d)


fl = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\annotation with id\\sentences.txt'
fileObj = open(fl, "r") #opens the file in read mode
sentence= fileObj.read().splitlines()

#print(sentence)


with open('C:\\Users\\pc\\PycharmProjects\\video-captioning\\annotation with id\\annotations_with_id.csv', 'a') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')

    for k,n in zip(id,sentence):
        #writer.writerow([k]+[n])
        output_row = [k]
        output_row.extend([n])
        writer.writerow(output_row)