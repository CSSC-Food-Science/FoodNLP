import csv

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
    return corpus, labels, codes