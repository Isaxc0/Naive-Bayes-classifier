import math

class NBClassifier:
    """
        Naive Bayes classifier
    """
    def __init__(self):
        pass

    #generates all unique words in training data
    def _set_known_vocabulary(self, training_data):
        known = []
        for doc, label in training_data:
            for word in doc.keys():
                known.append(word)
        self.known = set(known)

    #generates class priors
    def _set_priors(self, training_data):
        priors = {}
        labels = FreqDist([review[1] for review in training_data])
        total = 0
        for (label, val) in labels.items():
            labels[label] = val + 1
            total += 1
        total += len(training_data)
        for (label, val) in labels.items():
            priors[label] = val / total
        self.priors = priors

    #generates conditional probabilties for the training data
    def _set_cond_probs(self, training_data):
        conds = {}
        weather = [r[0] for r in training_data if r[1] == "weather"]
        football = [r[0] for r in training_data if r[1] == "football"]
        wlabels = FreqDist(reduce(lambda word_list, review: word_list + list(review.keys()), weather, []))
        flabels = FreqDist(reduce(lambda word_list, review: word_list + list(review.keys()), football, []))

        for word in known_vocabulary(training_data):
            wlabels[word] = wlabels.get(word, 0) + 1
            flabels[word] = flabels.get(word, 0) + 1

        wtotal = sum([count for (word, count) in wlabels.items()])
        ftotal = sum([count for (word, count) in flabels.items()])

        wprob = {}
        for word, count in w_labels.items():
            wprob[word] = count / wtotal
        conds["weather"] = wprob

        fprob = {}
        for ford, count in f_labels.items():
            fprob[ford] = count / ftotal
        conds["football"] = fprob

        self.conds = conds

    #calls appropraite functions to train classifier
    def train(self, training_data):
        self._set_known_vocabulary(training_data)
        self._set_priors(training_data)
        self._set_cond_probs(training_data)

    #classifies a document
    def classify(self, doc):
        doc_prob = {key: math.log(value) for (key, value) in self.priors.items()}
        for word in doc.keys():
            if word in self.known:
                for label, prob in doc_prob.items():
                    new_prob = math.log(self.conds[label].get(word, 0)) + prob
                    doc_prob.update({label: new_prob})
        doc_prob = sorted(doc_prob.items(), key=lambda x: x[1], reverse=True)
        if doc_prob[0][1] == doc_prob[1][1]:
            return random.choice([doc_prob[0], doc_prob[1]])
        return doc_prob[0]

    #classifies multiple documents
    def batch_classify(self, docs):
        return [self.classify(doc) for doc in docs]

