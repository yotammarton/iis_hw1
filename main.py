import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

"""Section 1.A."""


def word_extraction(sentence, stop_words):
    """
    :param sentence: pre-preprocessed sentence
    :param stop_words: nltk stopwords list
    :return: lower words of the given sentence, without stopwords
    """
    words = sentence.split()
    cleaned_text = [w.lower() for w in words if w.lower() not in stop_words]
    return cleaned_text


def tokenize(sentences, stop_words):
    """
    :param sentences: pre-preprocessed sentence
    :param stop_words: nltk stopwords list
    :return: vocabulary list
    """
    words = []
    for sentence in sentences:
        w = word_extraction(sentence, stop_words)
        words.extend(w)
        words = sorted(list(set(words)))
    return words


def load_data_to_list_of_greetings():
    """
    load the data and return a list of greetings as string. each word in the string is separated by space

    removes symbols
    remove sentences with only single word

    :return: list
    """

    with open('items.csv') as f:
        lines = f.readlines()

    all_lines = ''.join(lines)
    splitlines = all_lines.split(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n")
    splitlines = [line.replace('\n', ' ') for line in splitlines]
    splitlines = [line.replace('\'', '') for line in splitlines]
    splitlines = [re.sub(r'[^\w]', ' ', line).strip() for line in splitlines]
    splitlines = [re.sub(' +', ' ', line) for line in splitlines]  # remove extra spaces
    splitlines = [line for line in splitlines if line != '' and len(line.split(' ')) != 1]
    return splitlines


def generate_bow():
    """
    get the raw sentences, process them, create vocabulary
    :return: bag of words for each sentence (include counts), list of original sentences
    """
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  # get nltk stopwords
    # add stopwords without "'"
    stop_words = stop_words.union(set([stop_word.replace('\'', '') for stop_word in stop_words]))

    allsentences = load_data_to_list_of_greetings()
    vocab = tokenize(allsentences, stop_words)
    # print("Word List for Document \n{0} \n".format(vocab))

    bag_matrix = np.empty((0, len(vocab)), int)
    for sentence in allsentences:
        words = word_extraction(sentence, stop_words)
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
        bag_matrix = np.vstack([bag_matrix, bag_vector])
    return bag_matrix, allsentences


"""Section 1.B."""


def create_matrix():
    """
    :return: #ideas x # ideas matrix with counts of appearances in the same cluster
    """
    ideas_matrix = np.zeros((10, 10))

    with open('matrix_assignment.csv', 'r') as file:
        # datareader = csv.reader(file)
        for row in file:
            for list in row.split('[')[2:]:
                list = re.sub(r'["[\] ]', '', list)
                list = [int(i) for i in list.split(',') if i != '']
                for i in list:
                    for j in list:
                        ideas_matrix[i - 1][j - 1] += 1

    return ideas_matrix


def return_similar_and_unsimilar_idea(mat, idea1):
    """
    :param mat: appearances counts matrix
    :param idea1: given idea
    :return: similar and unsimilar ideas with respect to idea1
    """
    unsimilar_idea = mat[idea1 - 1].argmin() + 1
    tmp_row = np.array([val if idx != (idea1 - 1) else -np.inf for idx, val in enumerate(mat[idea1 - 1])])
    similar_idea = tmp_row.argmax() + 1
    return similar_idea, unsimilar_idea


if __name__ == '__main__':
    """Section 1.A."""
    # bag_matrix, allsentences = generate_bow()

    """Section 1.B."""
    ideas_matrix = create_matrix()
    df = DataFrame(ideas_matrix, index=range(1, 11, 1), columns=range(1, 11, 1))
    sns.heatmap(df, annot=False, cmap="YlGnBu", vmin=0, vmax=5000)
    plt.show()

    for idea in range(1, 11, 1):
        similar_idea, unsimilar_idea = return_similar_and_unsimilar_idea(ideas_matrix, idea)
        print(f'for idea {idea}: {similar_idea} is similar idea, {unsimilar_idea} is unsimilar idea')
