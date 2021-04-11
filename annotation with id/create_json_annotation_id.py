import argparse
import hashlib
import io
import json
import sys
import csv
from numpy import unicode


class Annotations:
    def __init__(self):
        self.annotations = []

        self.res = {
                    "type": 'captions',

                    "annotations": self.annotations,

                    }


    def read_multiple_files(self, filelist):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename)



    def read_file(self, sentence,d):
        for k,n in zip(sentence,d):

            self.annotations.append({
                "id": n,
                "caption": k,
            })

            #print(self.annotations)

    def dump_json(self, outfile):

        #self.res["annotations"] = self.annotations
        res = self.annotations
        print(res)
        with io.open(outfile, 'w') as fd:
            fd.write(unicode(json.dumps(res, ensure_ascii=True, sort_keys=True, indent=2, separators=(',', ': '))))


def main():
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

    fileObj = open(fileName, "r")  # opens the file in read mode
    id = fileObj.read().splitlines()

    # file = open('id.txt', "w+") #opens the file in read mode
    d = []
    for j in range(len(dates)):
        for i in range(len(id)):
            if (id[i] == scores[j]):
                d.append(dates[j])

        else:
            continue

    fl = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\annotation with id\\sentences.txt'
    fileObj = open(fl, "r")  # opens the file in read mode
    sentence = fileObj.read().splitlines()


    output_file="video_captioning.json"


    crf = Annotations()
    crf.read_file(sentence,d)
    crf.dump_json(output_file)
    print('Created json references in %s' % output_file)


if __name__ == "__main__":
    main()