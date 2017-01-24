from helper import Word2VecHelper, FileHelper
classes = []



if __name__ == '__main__':
    files = ["./train/dbpedia/generic/positive.txt", "./train/dbpedia/generic/negative.txt"]
    #Word2VecHelper.createModel(files, "dbpedia_generic")
    FileHelper.createTrainFile(FileHelper.categories, directory="train/dbpedia/generic", name="dbpedia_generic_train")
