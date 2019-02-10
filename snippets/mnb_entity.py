import pandas as pd


class Entity:

    df = None
    class_list = ['negative', 'positive', 'neutral']
    class_counts = pd.DataFrame({
        'Class': class_list,
        'Count': 0 * len(class_list)
    }).set_index('Class')

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
        self.class_counts.loc[review_class] += 1

    def unlearn_review(self, preprocessed_word_dict, review_class):

        for term, count in preprocessed_word_dict.items():
            # update counts for word in a class
            self.df.loc[review_class][term] -= count

        # decrement class count
        self.class_counts.loc[review_class] -= 1

    def print(self):
        print(self.df)
    
    # predict the class of the incoming review
    def predict(self, preprocessed_word_dict):

        print("USING ENTITY LEVEL CLASSIFIER")

        # sum of all term frequencies = class counts
        row_sum = self.df.sum(axis=1)
        prob = 1
        # get the probabilities for each class given data
        for term in preprocessed_word_dict:
            term_count = self.df.loc[self.class_list][term]
            prob *= term_count / row_sum

        # multiply with class priors
        prob *= self.class_counts['Count'] / self.class_counts['Count'].sum()
        return prob.idxmax()
