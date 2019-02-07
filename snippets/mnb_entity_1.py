import pandas as pd
import mnb_entity

# TO DO:
# def Get_class_counts
# def get total terms per entity
# How to determine top words?
# how to determine top entities
# wrapper to process data to provide to this method
# total number of reviews


class MultinomialNB:

    class_list = ['negative', 'positive', 'neutral']
    class_counts = {'negative': 0, 'positive': 0, 'neutral': 0}

    # this dictionary will contain all the entities
    global_entity_dictionary = {}

    # global dataframe which will contain all terms frequencies, all classes in 1 dataframe for all the
    # entities, use this dataframe to predict for a new entity that has no history
    global_term_frequency = pd.DataFrame({'Class': class_list}).set_index('Class')

    # update counts in data frame when encountering an existing product ID
    def learn(self, entity_id, preprocessed_word_dict, review_class):

        # if new entity, create a new Entity object and add the new object to global_entity_dictionary
        if entity_id not in self.global_entity_dictionary.keys():
            self.global_entity_dictionary[entity_id] = mnb_entity.Entity(preprocessed_word_dict, self.class_list)

        # update global class counts
        self.class_counts[review_class] += 1

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
        self.class_counts[review_class] -= 1

        # update global term frequency
        for term, count in preprocessed_word_dict.items():
            # update counts for word in a class
            self.global_term_frequency.loc[review_class][term] -= count

        # remove the example from the model
        self.global_entity_dictionary[entity_id].unlearn_review(preprocessed_word_dict, review_class)


if __name__ == '__main__':

    entity_id1 = 'B00FRH4F66'
    entity_id2 = 'B00AWMBZR8'
    processed_word_dict1 = {
        'today': 1,
        '-PRON-': 4,
        'get': 1,
        'like': 1,
        'nice': 1,
        'length': 1,
        'think': 1,
        'probably': 1,
        'heavy': 1,
        'but': 1,
        'actually': 1,
        'lite': 1,
        'weight': 1,
        'nice length': 1,
        '-PRON- think': 1,
        'actually lite': 1,
        'lite weight': 1,
        '-PRON- just get': 1,
        'get -PRON- today': 1,
        'today and -PRON-': 1,
        '-PRON- really like': 1,
        '-PRON- a nice': 1,
        'probably be heavy': 1,
        'but -PRON- actually': 1,
        'actually lite weight': 1
    }

    processed_word_dict2 = {
        'daughter': 1,
        'love': 1,
        'retro': 1,
        'fit': 1,
        'bill': 1,
        'perfectly': 1,
        'daughter love': 1,
        'love retro': 1,
        'bill perfectly': 1,
        'daughter love retro': 1,
        'fit the bill': 1
    }

    mnb = MultinomialNB()

    mnb.learn(entity_id1, processed_word_dict1, 'positive')
    mnb.learn(entity_id1, processed_word_dict1, 'negative')

    mnb.learn(entity_id2, processed_word_dict1, 'positive')
    mnb.learn(entity_id2, processed_word_dict2, 'negative')

    print("BEFORE")
    mnb.global_entity_dictionary[entity_id1].print()
    print(mnb.global_entity_dictionary[entity_id1].class_counts)
    print('global term freq: \n', mnb.global_term_frequency)
    print("BEFORE")

    mnb.unlearn(entity_id2, processed_word_dict2, 'negative')
    mnb.unlearn(entity_id1, processed_word_dict1, 'negative')

    print("AFTER")
    mnb.global_entity_dictionary[entity_id1].print()
    print(mnb.global_entity_dictionary[entity_id1].class_counts)
    print('global term freq: \n', mnb.global_term_frequency)
    print("AFTER")

'''
# for every word in the dictionary, add a column to dataframe, if term doesnt exist
if operation == 'learn':
    for term, count in preprocessed_word_dict.items():
        # add a term that is not existing
        if term not in self.global_entity_dictionary[entity_id]:
            self.global_entity_dictionary[entity_id][term] = 0
        # update counts for word in a class
        self.global_entity_dictionary[entity_id].loc[review_class][term] += count

        # update the global term frequency dataframe

    # increment class count
    self.class_counts[review_class] += 1

elif operation == 'unlearn':
    for term, count in preprocessed_word_dict.items():
        # update counts for word in a class
        self.global_entity_dictionary[entity_id].loc[review_class][term] -= count

    # decrement class count
    self.class_counts[review_class] -= 1
'''