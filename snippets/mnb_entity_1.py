import pandas as pd
import snippets.mnb_entity as entity
import random


# TO DO:
# def Get_class_counts
# def get total terms per entity
# How to determine top words?
# how to determine top entities
# wrapper to process data to provide to this method
# total number of reviews


class MultinomialNB:

    class_list = ['negative', 'positive', 'neutral']
    class_counts = pd.DataFrame({
        'Class': class_list,
        'Count': 0 * len(class_list)
    }).set_index('Class')

    # this dictionary will contain all the entities
    global_entity_dictionary = {}

    # global dataframe which will contain all terms frequencies, all classes in 1 dataframe for all the
    # entities, use this dataframe to predict for a new entity that has no history
    global_term_frequency = pd.DataFrame({'Class': class_list}).set_index('Class')

    # update counts in data frame when encountering an existing product ID
    def learn(self, entity_id, preprocessed_word_dict, review_class):

        # if new entity, create a new Entity object and add the new object to global_entity_dictionary
        if entity_id not in self.global_entity_dictionary.keys():
            self.global_entity_dictionary[entity_id] = entity.Entity(preprocessed_word_dict, self.class_list)

        # update global class counts
        self.class_counts.loc[review_class] += 1

        # update global term frequency
        for term, count in preprocessed_word_dict.items():
            # add a term that is not existing
            if term not in self.global_term_frequency:
                self.global_term_frequency[term] = 0
            # update counts for word in a class
            self.global_term_frequency.loc[review_class][term] += count

        # add the example to the model
        self.global_entity_dictionary[entity_id].learn(preprocessed_word_dict, review_class)

    def unlearn(self, entity_id, preprocessed_word_dict, review_class):

        # if entity does not exist yet, raise an exception
        if entity_id not in self.global_entity_dictionary.keys():
            raise ValueError("entity_id ", entity_id, " is not present in the model.")

        # update global class counts
        self.class_counts.loc[review_class] -= 1

        # update global term frequency
        for term, count in preprocessed_word_dict.items():
            # update counts for word in a class
            self.global_term_frequency.loc[review_class][term] -= count

        # remove the example from the model
        self.global_entity_dictionary[entity_id].unlearn_review(preprocessed_word_dict, review_class)

    # predict the class of the incoming review
    def predict(self, entity_id, preprocessed_word_dict):

        if entity_id in self.global_entity_dictionary:
            return self.global_entity_dictionary[entity_id].predict(preprocessed_word_dict)

        print("USING GLOBAL CLASSIFIER")

        print(self.global_term_frequency)

        print(self.class_counts)

        # sum of all term frequencies = class counts
        row_sum = self.global_term_frequency.sum(axis=1)
        prob = 1
        # get the probabilities for each class given data
        for term in preprocessed_word_dict:

            term_count = self.global_term_frequency.loc[self.class_list][term]
            prob *= term_count / row_sum

        # multiply with class priors
        prob *= self.class_counts['Count'] / self.class_counts['Count'].sum()
        return prob.idxmax()


if __name__ == '__main__':

    entity_id1 = 'B00FRH4F66'
    entity_id2 = 'B00AWMBZR8'
    entity_id3 = 'B00AWMBZR9'
    processed_word_dict1 = {
        'horrible': 30,
        'bad': 50,
        'wonderful': 1,
        'awesome': 1
    }

    processed_word_dict2 = {
        'horrible': 1,
        'bad': 1,
        'wonderful': 50,
        'awesome': 20
    }

    processed_word_dict3 = {
        'horrible': 1,
        'bad': 1,
        'wonderful': 50,
        'awesome': 30
    }

    mnb = MultinomialNB()

    #mnb.learn(entity_id1, processed_word_dict1, 'positive')
    #mnb.learn(entity_id1, processed_word_dict1, 'negative')

    #mnb.learn(entity_id2, processed_word_dict2, 'positive')
    #mnb.learn(entity_id2, processed_word_dict2, 'negative')

    #print("BEFORE")
    #mnb.global_entity_dictionary[entity_id1].print()
    #print(mnb.global_entity_dictionary[entity_id1].class_counts)
    #print('global term freq: \n', mnb.global_term_frequency)
    #print("BEFORE")

    #mnb.unlearn(entity_id2, processed_word_dict2, 'negative')
    #mnb.unlearn(entity_id1, processed_word_dict1, 'negative')

    #print("AFTER")
    #mnb.global_entity_dictionary[entity_id1].print()
    #print(mnb.global_entity_dictionary[entity_id1].class_counts)
    #print('global term freq: \n', mnb.global_term_frequency)
    #print("AFTER")

    #pred = mnb.predict(entity_id3, processed_word_dict3)
    #print("\nPREDICTION IS: ", pred)

    # getting the data in 3 column format by resetting the multi-index

    data = pd.read_pickle(
        "/Users/amused_confused/Documents/OVGU/Hiwi/osm/data/amazon/filtered/weekly/converted/20140727.pkl.gzip")

    # flatten the MultiIndex
    data.reset_index(inplace=True)
    # extract the entity Id
    data['entity_id'] = data['review_id'].str.split("_").str[1]
    # keep only the required columns
    data = data[['entity_id', 'ngrams', 'stars']]
    print(data)

    for index, row in data.iterrows():
        mnb.learn(row.entity_id, row.ngrams, row.stars)

    pred = []
    for index, row in data.iterrows():
        pred.append(mnb.predict(row.entity_id, row.ngrams))

    print(pred)
