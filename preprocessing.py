from helper import Word2VecHelper, FileHelper
import os
classes = []


def create(ontology, type):
    train_file = "{}_{}_train.tsv".format(ontology,type)
    test_file = "{}_{}_test.tsv".format(ontology, type)
    if not os.path.exists(train_file):
        FileHelper.createTrainFile(FileHelper.categories, directory="train/{}/{}".format(ontology,type), name=train_file)
    if not os.path.exists(test_file):
        FileHelper.createTrainFile(FileHelper.categories, directory="test/{}/{}".format(ontology,type), name=test_file)

    return train_file,test_file



if __name__ == '__main__':
    files = ["./train/dbpedia/generic/positive.txt", "./train/dbpedia/generic/negative.txt"]
    #Word2VecHelper.createModel(files, "dbpedia_generic")
    print(create("yago", "specific"))

