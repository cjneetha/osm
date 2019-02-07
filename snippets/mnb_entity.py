import pandas as pd


class Entity:

    df = None
    class_counts = {'negative': 0, 'positive': 0, 'neutral': 0}

    # create a new data frame when encountered a new product ID
    def __init__(self, preprocessed_word_dict, class_list):

        # create empty data frame with class labels as index column
        self.df = pd.DataFrame({'Class': class_list}).set_index('Class')
        # read all words to make column list
        column_list = [key for key in preprocessed_word_dict.keys()]
        # create columns with the words and set them to 0
        for column in column_list:
            self.df[column] = 0

    def learn(self, preprocessed_word_dict, review_class):

        for term, count in preprocessed_word_dict.items():
            # add a term that is not existing
            if term not in self.df:
                self.df[term] = 0
            # update counts for word in a class
            self.df.loc[review_class][term] += count

        # increment class count
        self.class_counts[review_class] += 1

    def unlearn_review(self, preprocessed_word_dict, review_class):

        for term, count in preprocessed_word_dict.items():
            # update counts for word in a class
            self.df.loc[review_class][term] -= count

        # decrement class count
        self.class_counts[review_class] -= 1

    def print(self):
        print(self.df)
