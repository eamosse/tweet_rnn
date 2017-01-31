from helper import Word2VecHelper, FileHelper
import os
import glob
from helper.nerd import NERD
classes = []


def concat(folder, ontology, type, nclass):
    classes = FileHelper.categories if nclass == 8 else FileHelper.binaries
    train_file = "{}_{}_{}_{}.tsv".format(ontology, type, nclass,folder)
    with open(train_file, 'w') as writer:
        for f in glob.glob("{}/{}/{}/*.txt".format(folder, ontology, type)):
            if f.split("/")[len(f.split("/")) - 1].split(".")[0] in classes:
                with open(f) as ff:
                    for line in ff:
                        if not line.isspace() and len(line.split("\t")) == 3:
                            writer.write(line.replace('\n', '')+"\n")
                        else:
                            print(line, 'is space')
    return train_file

def create(ontology, type, nclass=8):
    train_file = concat("train", ontology,type,nclass)
    test_file = concat("test", ontology,type,nclass)

    return train_file,test_file

if __name__ == '__main__':
    #FileHelper.generateDataFile()
    create("dbpedia", "generic")