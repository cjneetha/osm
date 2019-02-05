import pandas as pd

# TO DO:
# def Get_class_counts
# def get total terms per entity
# How to determine top words?
# how to determine top entities
# wrapper to process data to provide to this method
# total number of reviews


class MultinomialNB:

    class_list = ['negative', 'positive', 'neutral']
    global_entity_dictionary = {}
    global_term_frequency = pd.DataFrame
    class_counts = {'negative': 0, 'positive': 0, 'neutral': 0}
    learning_mode_flag = 0

    # create 1 global dataframe which will contain all terms frequencies, all classes  in 1 dataframe for all the
    # entities, use this dataframe to predict for a new entity that has no history
    def create_global_dataframe(self):
        return pd.DataFrame({'Class': self.class_list}).set_index('Class')

    # def __init__(self):
    # read the class Series, get all unique values, initialize in class list

    # create a new data frame when encountered a new product ID
    def create_new_df(self, entity_id, preprocessed_word_dict):
        # create empty data frame with class labels as index column
        df = pd.DataFrame({'Class': self.class_list}).set_index('Class')

        # read all words to make column list
        column_list = [key for key in preprocessed_word_dict.keys()]
        # create columns with the words and set them to 0
        for column in column_list:
            df[column] = 0

        # assign the new data frame to the global dictionary
        self.global_entity_dictionary[entity_id] = df

    # update counts in data frame when encountered a existing product ID
    def update_existing_df(self, entity_id, preprocessed_word_dict, review_class, operation='learn'):
        # for every word in the dictionary, add a column to dataframe, if term doesnt exist
        if operation == 'learn':
            for word, count in preprocessed_word_dict.items():
                # add a term that is not existing
                if word not in self.global_entity_dictionary[entity_id]:
                    self.global_entity_dictionary[entity_id][word] = 0
                # update counts for word in a class
                self.global_entity_dictionary[entity_id].loc[review_class][word] += count

            # increment class count
            self.class_counts[review_class] += 1

        elif operation == 'unlearn':
            for word, count in preprocessed_word_dict.items():
                # update counts for word in a class
                self.global_entity_dictionary[entity_id].loc[review_class][word] -= count
                # decrement class count
            self.class_counts[review_class] -= 1



if __name__ == '__main__':

    mnb = MultinomialNB()
    # print(type(mnb.global_entity_dictionary))

    # mnb.fit(X_train, y_train)

    entity_id1 = 'B00FRH4F66'
    entity_id2 = ''
    processed_word_dict = {
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
         'actually lite weight': 1}

    processed_word_dict2 = {'daughter': 1,
 'love': 1,
 'retro': 1,
 'fit': 1,
 'bill': 1,
 'perfectly': 1,
 'daughter love': 1,
 'love retro': 1,
 'bill perfectly': 1,
 'daughter love retro': 1,
 'fit the bill': 1}

    mnb.create_new_df(entity_id1, processed_word_dict)
    mnb.update_existing_df(entity_id1,processed_word_dict,'positive')
    mnb.update_existing_df(entity_id1,processed_word_dict,'negative')


    mnb.create_new_df('B00AWMBZR8',processed_word_dict)
    mnb.update_existing_df('B00AWMBZR8',processed_word_dict,'positive')
    mnb.update_existing_df(entity_id1,processed_word_dict2,'negative')
    mnb.update_existing_df('B00AWMBZR8',processed_word_dict2,'negative')
    print("BEFORE")
    print(mnb.global_entity_dictionary['B00FRH4F66'])
    print(mnb.class_counts)
    print("BEFORE")
    mnb.update_existing_df(entity_id1,processed_word_dict2,'negative', 'unlearn')


    print("AFTER")
    print(mnb.class_counts)
    print(mnb.global_entity_dictionary['B00FRH4F66'])
    print("AFTER")



