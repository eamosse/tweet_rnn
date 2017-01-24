from helper import Word2VecHelper, FileHelper
classes = []



if __name__ == '__main__':
    files = ["./test/dbpedia/generic/positive.txt", "./test/dbpedia/generic/negative.txt"]
    #Word2VecHelper.createModel(files, "dbpedia_generic")
    FileHelper.createTrainFile(FileHelper.categories, directory="test/dbpedia/generic", name="dbpedia_generic_test")
