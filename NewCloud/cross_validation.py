from sklearn.model_selection import cross_val_score

def CrossValidation(clf, X, y):
    scores = cross_val_score(clf, X, y, cv = 10, scoring = 'accuracy')
    media = scores.mean()
    print(scores, media)
