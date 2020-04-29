import csv
additional = "./data/additional_data.csv"
def preprocessGPQI(filename):
    codes = set()
    col_names = ""
    labels = []
    corpus = []
    with open(filename) as cv:
        reader = csv.reader(cv, delimiter=',')
        first_line = True
        for row in reader:
            if first_line:
                col_names = ", ".join(row)
                first_line = False
            else:
                code = row[2]
                if not code.isnumeric() or len(row[1]) == 0: # empty or NAN
                    continue
                code = int(code)
                codes.add(code)
                labels.append(code)
                corpus.append(row[1].lower())
    print("Size without add: labels("+ str(len(labels)) + ") corpus(" + str(len(corpus)) + ")")
    if (additional != None):
        with open(additional) as av:
            reader = csv.reader(av, delimiter=',')
            first_line = True
            for row in reader:
                if first_line:
                    col_names = ", ".join(row)
                    first_line = False
                else:
                    code = row[8]
                    if not code.isnumeric() or len(row[8]) == 0: # empty or NAN
                        continue
                    code = int(code)
                    codes.add(code)
                    labels.append(code)
                    corpus.append(row[4].lower())
    print("Size with add: labels("+ str(len(labels)) + ") corpus(" + str(len(corpus)) + ")")
    return corpus, labels, codes