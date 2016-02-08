import pandas as pd
import pickle

def note_submission_info(msg, file_name):
    f = open("submission_history.txt", 'a')

    f.write(file_name + ":\n")
    f.write(msg + "\n\n")

    f.close()


def build_submission(clf, X_test, ids, target_file="submission.csv"):
    try:
        y_submission = clf.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_submission = clf.predict(X_test)

    df_result = pd.DataFrame({"ID": ids, "PredictedProb": y_submission})
    df_result.to_csv(target_file, index=False)

    pickle.dump(clf, open(target_file.replace(".csv", ".model"), "wb"))
