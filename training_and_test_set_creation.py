from Bio import SeqIO
import random


def IDs_test_and_training_set (file): 
    every_records = []
    positive_records = []
    negative_records = []

    # Get the records of all allergens and non-allergens
    for record in SeqIO.parse(file,'fasta'):
        every_records.append(record)
        if record.id.startswith('P') :
            positive_records.append(record.id)
        elif record.id.startswith('N') :
            negative_records.append(record.id)
        else :
            print(f'ERROR : protein {record.id}')

    # Number of protein in Test set
    nb_positive_training = round(len(positive_records)*0.8)
    nb_negative_training = round(len(negative_records)*0.8)

    # Number of protein in Training set
    nb_positive_test = len(positive_records)-nb_positive_training
    nb_negative_test = len(negative_records)-nb_negative_training

    # IDs of proteins in Test set
    test_set_positive = random.sample(positive_records, nb_positive_test)
    test_set_negative = random.sample(negative_records, nb_negative_test)
    test_set = test_set_positive + test_set_negative
    print('Size of the test set:',len(test_set))

    # IDs of proteins in Training set 
    training_set_positive = positive_records.copy()
    training_set_negative = negative_records.copy()

    for protein in test_set :
        if protein in positive_records :
            training_set_positive.remove(protein)
        elif protein in negative_records : 
            training_set_negative.remove(protein)

    training_set = training_set_positive + training_set_negative
    print('Size of the training set:',len(training_set))

    return training_set, test_set, nb_positive_training, training_set_positive, training_set_negative



def verification (training_set, test_set) :
    """ To verify that no protein in test set is in training set
        and vice versa """
    correct = True
    
    for protein in test_set :
        if protein in training_set :
            print(f"ERROR — {protein} from test set in Training set")
            correct = False
    
    for protein in training_set :
        if protein in test_set :
            print(f"ERROR — {protein} from Training set in test set")
            correct = False 
    return correct


def create_fasta_training_and_test (file, training_set, test_set, training_set_file, test_set_file):
    """ Creation of the fasta files for the training and the test set
        by getting the records in the complete filtered database """
    
    training_records = []
    test_records = []
    
    for record in SeqIO.parse(file, 'fasta') :
        if record.id in training_set :
            training_records.append(record)
        elif record.id in test_set :
            test_records.append(record)
        else :
            print(f"ERROR — {record.id} not found in training set or test set")

    
    with open(training_set_file, 'w') as fasta_training :
        SeqIO.write(training_records, fasta_training, 'fasta')
    with open(test_set_file, 'w') as fasta_test :
        SeqIO.write(test_records, fasta_test, 'fasta')


def create_undersampling_training_set(file, nb_positive_training, training_set_positive, training_set_negative, undersamp_training_set_file):
    """ Creation of the undersampled training set
        -> same number of Positives and Negatives
    """
    
    undersamp_negatives = random.sample(training_set_negative, nb_positive_training)
    undersamp_records = []
    
    for record in SeqIO.parse(file, 'fasta') :
        if record.id in training_set_positive or record.id in undersamp_negatives :
            undersamp_records.append(record)
        
    
    print('Size of the undersampled training set :', len(undersamp_records))
    
    with open(undersamp_training_set_file, 'w') as fasta_undersamp :
        SeqIO.write(undersamp_records, fasta_undersamp, 'fasta')



if __name__ == "__main__" :
    
    file = 'complete_db_40.txt' # Complete filtered database
    undersamp_training_set_file = 'Undersamp_training_set.txt' # Name of the undersampled training set to create
    training_set_file = 'Training_set.txt' # Name of the training set to create
    test_set_file = 'Test_set.txt' # Name of the test set to create
    
    
    random.seed(42)
    training_set, test_set, nb_positive_training, training_set_positive, training_set_negative = IDs_test_and_training_set(file)
    correct = verification(training_set, test_set)
    
    if correct :
        create_fasta_training_and_test(file, training_set, test_set, training_set_file, test_set_file)
        create_undersampling_training_set(file, nb_positive_training, training_set_positive, training_set_negative, undersamp_training_set_file)
        
    