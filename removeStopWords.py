from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_DIR, "sample_data", "Forum")
TRAINING_FILE = ("forumTraining.data.txt", "forumTraining-clean.data.txt")
TEST_FILE =  ("forumTest.data.txt", "forumTest-clean.data.text")


def clean():
    stop_words = set(stopwords.words('english')) 
    for origin, new in [TRAINING_FILE, TEST_FILE]:
        origin_file = os.path.join(BASE_PATH, origin)
        new_file = os.path.join(BASE_PATH, new)

        W = ""
        with open(origin_file, "r") as f:
            for line in f:
                l = ""
                category, document = line.split(maxsplit=1)
                words = word_tokenize(document)
                for w in words:
                    if not w in stop_words and len(w) > 3:
                        l += w + " "
                l += "\n"

                W += category + " " + l
        f.close()
        dst = open(new_file, "w")
        dst.write(W)
        dst.close()

if __name__ == "__main__":
    clean()


