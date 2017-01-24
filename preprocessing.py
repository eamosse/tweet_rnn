from helper import Word2VecHelper, FileHelper
import os
from helper.nerd import NERD
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
    """
    from SPARQLWrapper import SPARQLWrapper, JSON

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE { <http://dbpedia.org/resource/Asturias> rdfs:label ?label }
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["label"]["value"])
    """

    """
     params = []
    texts = ["Just commented on @thejournal_ie: Penn State scandal: coach Sandusky jailed for life for child abuse - http://jrnl.ie/628763", "Just commented on @thejournal_ie: Penn State scandal: coach Sandusky jailed for life for child abuse - http://jrnl.ie/628763 "]
    index = 0
    separator = "==="
    tt = ""
    for text in texts:
        params.append({"start":index, "end":len(text)+index+len(separator), 'text':text, 'annotations': []})
        index+=len(text)+len(separator)
        tt+=text+separator

    print(params)

    timeout = 10
    n = NERD('nerd.eurecom.fr', "cci7nqeutkegjsjlb4rrobd9s88vdejl")
    data = n.extract(tt, 'combined', timeout)
    print(data)
    for d in data:
        for index, t in enumerate(params):
            print(index)
            if d['startChar'] >= t['start'] and d['endChar'] <= t['end']:
                t['annotations'].append(d)
                #params.remove(index)
                break


    print(data)
    print(params)

    """
    data = [{'relevance': 0.5, 'uri': 'http://dbpedia.org/resource/TheJournal.ie', 'idEntity': 25517802, 'extractorType': 'http://dbpedia.org/ontology/Newspaper,http://dbpedia.org/ontology/PeriodicalLiterature,http://dbpedia.org/ontology/WrittenWork,http://dbpedia.org/ontology/Work', 'confidence': 0.8609, 'label': 'thejournal_ie', 'startChar': 19, 'endChar': 32, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'dandelionapi'}, {'relevance': 0.5, 'uri': 'http://en.wikipedia.org/wiki/Penn_State_child_sex_abuse_scandal', 'idEntity': 25517803, 'extractorType': '', 'confidence': 0.791649, 'label': 'Penn State scandal', 'startChar': 34, 'endChar': 52, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'zemanta'}, {'relevance': 0.5, 'uri': '', 'idEntity': 25517804, 'extractorType': None, 'confidence': -1.0, 'label': 'Sandusky', 'startChar': 60, 'endChar': 68, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'thd'}, {'relevance': 0.5, 'uri': 'http://en.wikipedia.org/wiki/Life_imprisonment', 'idEntity': 25517805, 'extractorType': '', 'confidence': 0.645517, 'label': 'jailed for life', 'startChar': 69, 'endChar': 84, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'zemanta'}, {'relevance': 0.0, 'uri': 'http://en.wikipedia.org/wiki/Child_abuse', 'idEntity': 25517806, 'extractorType': '/tv/tv_subject,/book/book_subject,/people/cause_of_death,/medicine/risk_factor,/organization/organization_sector,/film/film_subject', 'confidence': 0.152842, 'label': 'child abuse', 'startChar': 89, 'endChar': 100, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}, {'relevance': 0.0, 'uri': '', 'idEntity': 25517807, 'extractorType': 'URL', 'confidence': 0.0, 'label': 'http://jrnl.ie/628763===Just', 'startChar': 103, 'endChar': 131, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}, {'relevance': 0.5, 'uri': 'http://dbpedia.org/resource/TheJournal.ie', 'idEntity': 25517808, 'extractorType': 'http://dbpedia.org/ontology/Newspaper,http://dbpedia.org/ontology/PeriodicalLiterature,http://dbpedia.org/ontology/WrittenWork,http://dbpedia.org/ontology/Work', 'confidence': 0.8609, 'label': 'thejournal_ie', 'startChar': 146, 'endChar': 159, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'dandelionapi'}, {'relevance': 0.5, 'uri': 'http://dbpedia.org/resource/Penn_State_child_sex_abuse_scandal', 'idEntity': 25517809, 'extractorType': '', 'confidence': 0.8719, 'label': 'Penn State scandal', 'startChar': 161, 'endChar': 179, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'dandelionapi'}, {'relevance': 0.5, 'uri': '', 'idEntity': 25517810, 'extractorType': None, 'confidence': -1.0, 'label': 'Sandusky', 'startChar': 187, 'endChar': 195, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'thd'}, {'relevance': 0.0, 'uri': 'http://en.wikipedia.org/wiki/Child_abuse', 'idEntity': 25517811, 'extractorType': '/tv/tv_subject,/book/book_subject,/people/cause_of_death,/medicine/risk_factor,/organization/organization_sector,/film/film_subject', 'confidence': 0.152842, 'label': 'child abuse', 'startChar': 216, 'endChar': 227, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}, {'relevance': 0.0, 'uri': '', 'idEntity': 25517812, 'extractorType': 'URL', 'confidence': 0.0, 'label': 'http://jrnl.ie/628763\xa0===', 'startChar': 230, 'endChar': 255, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}]

    params = [{'Just commented on @thejournal_ie: Penn State scandal: coach Sandusky jailed for life for child abuse - http://jrnl.ie/628763': 'Just commented on @thejournal_ie: Penn State scandal: coach Sandusky jailed for life for child abuse - http://jrnl.ie/628763', 'end': 127, 'annotations': [{'relevance': 0.5, 'uri': 'http://dbpedia.org/resource/TheJournal.ie', 'idEntity': 25517802, 'extractorType': 'http://dbpedia.org/ontology/Newspaper,http://dbpedia.org/ontology/PeriodicalLiterature,http://dbpedia.org/ontology/WrittenWork,http://dbpedia.org/ontology/Work', 'confidence': 0.8609, 'label': 'thejournal_ie', 'startChar': 19, 'endChar': 32, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'dandelionapi'}, {'relevance': 0.5, 'uri': 'http://en.wikipedia.org/wiki/Penn_State_child_sex_abuse_scandal', 'idEntity': 25517803, 'extractorType': '', 'confidence': 0.791649, 'label': 'Penn State scandal', 'startChar': 34, 'endChar': 52, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'zemanta'}, {'relevance': 0.5, 'uri': '', 'idEntity': 25517804, 'extractorType': None, 'confidence': -1.0, 'label': 'Sandusky', 'startChar': 60, 'endChar': 68, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'thd'}, {'relevance': 0.5, 'uri': 'http://en.wikipedia.org/wiki/Life_imprisonment', 'idEntity': 25517805, 'extractorType': '', 'confidence': 0.645517, 'label': 'jailed for life', 'startChar': 69, 'endChar': 84, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'zemanta'}, {'relevance': 0.0, 'uri': 'http://en.wikipedia.org/wiki/Child_abuse', 'idEntity': 25517806, 'extractorType': '/tv/tv_subject,/book/book_subject,/people/cause_of_death,/medicine/risk_factor,/organization/organization_sector,/film/film_subject', 'confidence': 0.152842, 'label': 'child abuse', 'startChar': 89, 'endChar': 100, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}], 'start': 0}, {'annotations': [{'relevance': 0.5, 'uri': 'http://dbpedia.org/resource/TheJournal.ie', 'idEntity': 25517808, 'extractorType': 'http://dbpedia.org/ontology/Newspaper,http://dbpedia.org/ontology/PeriodicalLiterature,http://dbpedia.org/ontology/WrittenWork,http://dbpedia.org/ontology/Work', 'confidence': 0.8609, 'label': 'thejournal_ie', 'startChar': 146, 'endChar': 159, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'dandelionapi'}, {'relevance': 0.5, 'uri': 'http://dbpedia.org/resource/Penn_State_child_sex_abuse_scandal', 'idEntity': 25517809, 'extractorType': '', 'confidence': 0.8719, 'label': 'Penn State scandal', 'startChar': 161, 'endChar': 179, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'dandelionapi'}, {'relevance': 0.5, 'uri': '', 'idEntity': 25517810, 'extractorType': None, 'confidence': -1.0, 'label': 'Sandusky', 'startChar': 187, 'endChar': 195, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'thd'}, {'relevance': 0.0, 'uri': 'http://en.wikipedia.org/wiki/Child_abuse', 'idEntity': 25517811, 'extractorType': '/tv/tv_subject,/book/book_subject,/people/cause_of_death,/medicine/risk_factor,/organization/organization_sector,/film/film_subject', 'confidence': 0.152842, 'label': 'child abuse', 'startChar': 216, 'endChar': 227, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}, {'relevance': 0.0, 'uri': '', 'idEntity': 25517812, 'extractorType': 'URL', 'confidence': 0.0, 'label': 'http://jrnl.ie/628763\xa0===', 'startChar': 230, 'endChar': 255, 'nerdType': 'http://nerd.eurecom.fr/ontology#Thing', 'extractor': 'textrazor'}], 'end': 255, 'Just commented on @thejournal_ie: Penn State scandal: coach Sandusky jailed for life for child abuse - http://jrnl.ie/628763\xa0': 'Just commented on @thejournal_ie: Penn State scandal: coach Sandusky jailed for life for child abuse - http://jrnl.ie/628763\xa0', 'start': 127}]


    for p in params:
        p['annotations'] = []


    for p in params:
        annotations = [a for a in data if a['startChar'] >= p['start'] and a['endChar'] <= p['end']]
        p['annotations'] = annotations

    print(params)






