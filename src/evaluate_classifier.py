class ConfusionMatrix:
    def __init__(self, predictions, correct, classes=("P", "N")):
        (self.c1, self.c2) = classes
        self.TP = 0 #true positive
        self.FP = 0 #false positive
        self.FN = 0 #false negative
        self.TN = 0 #true negative
        for p, g in zip(predictions, correct):
            if g == self.c1:
                if p == self.c1:
                    self.TP += 1
                else:
                    self.FN += 1

            elif p == self.c1:
                self.FP += 1
            else:
                self.TN += 1

    #precision of classifier
    def precision(self):
        p = (self.TP) / (self.TP + self.FN)
        return p

    #recall of classifier
    def recall(self):
        r = (self.TP) / (self.TP + self.FN)
        return r

    #f1 score of classifier
    def f1(self):
        r = self.recall()
        p = self.precision()
        f1 = (2 * p * r) / (r + p)
        return f1