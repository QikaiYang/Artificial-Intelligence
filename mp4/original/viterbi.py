"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math
import numpy as np

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #A map to count the number of appearances of tags for each word
    count = {}   #count[word] = {tag1 : num1, tag2 : num2}
    for sentence in train:
        for word in sentence:
            if word[0] not in count:
                count[word[0]] = {}
                count[word[0]][word[1]] = 1
            else:
                if word[1] not in count[word[0]]:
                    count[word[0]][word[1]] = 1
                else:
                    count[word[0]][word[1]] += 1
    #-------------------------make predications---------------------------
    predicts = []
    for sentence in test:
        temp = []
        for word in sentence:
            if word not in count:
                temp.append((word, "NOUN"))
            else:
                temp.append((word, max(count[word],key=count[word].get)))    
        predicts.append(temp)
    return predicts

#----------------------------------------------------------------------------------------------------------------------
def get_hapax(emission, record_tags):
    result = {}  #[word][tag]
    for tag in emission:
        for word in emission[tag]:
            if emission[tag][word] == 1:
                if tag in result:
                    result[tag] += 1
                else:
                    result[tag] = 1
    all = sum(result.values())
    for tag in record_tags:
        if tag in result:
            result[tag] = result[tag]/all
        else:
            result[tag] = 0.000001
    return result

def smooth(map, factor):
    all = sum(map.values())
    for key in map:
        all +=  factor
        map[key] += factor
    for key in map:
        map[key] /= all
        map[key] = math.log(map[key])
    return map

def smooth_part2(map, factor, hapax, tag): #emission[tag][word] key is word
    all = sum(map.values())
    for key in map:
        all += factor*hapax[tag]
        map[key] += factor*hapax[tag]
    for key in map:
        map[key] /= all
        map[key] = math.log(map[key])
    return map

def get_initial(initial_count, tag):
    if tag in initial_count:
        return initial_count[tag]
    else:
        return initial_count["unknown"]

def get_emission(emission, tag, word):
    if word in emission[tag]:
        return emission[tag][word]
    else:
        return emission[tag]["unknown"]

def get_transfer(transfer, tag1, tag2):
    if tag2 in transfer[tag1]:
        return transfer[tag1][tag2]
    else:
        return transfer[tag1]["unknown"]

def viterbi(train, test):
    '''
    TODO: implement the Viterbi algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #-------------------------construct model-----------------------------        
    transfer = {}  
    emission = {}  
    initial_count = {"unknown":0}   
    #-------------------------construct emission--------------------------
    for sentence in train:
        for word in sentence:
            if word[1] in emission:
                if word[0] in emission[word[1]]:
                    emission[word[1]][word[0]] += 1
                else:
                    emission[word[1]][word[0]] = 1
            else:
                emission[word[1]] = {}
                emission[word[1]][word[0]] = 1
    #---------------------construct transfer & initial--------------------
    for sentence in train:
        if(sentence[0][1] not in initial_count):
            initial_count[sentence[0][1]] = 1
        else:
            initial_count[sentence[0][1]] += 1
        for num in range(len(sentence)-1):
            curr_tag = sentence[num][1]
            next_tag = sentence[num+1][1]
            if curr_tag in transfer:
                if next_tag in transfer[curr_tag]:
                    transfer[curr_tag][next_tag] += 1
                else:
                    transfer[curr_tag][next_tag] = 1
            else:
                transfer[curr_tag] = {}
                transfer[curr_tag][next_tag] = 1
    #-------------------------formalize everything------------------------
    record_tags = list(emission.keys())
    hapax = get_hapax(emission, record_tags)
    factor = 0.000009
    initial_count = smooth(initial_count, factor)
    for tag in record_tags:
        emission[tag]["unknown"] = 0
        emission[tag] = smooth(emission[tag], factor)#, hapax, tag)
        transfer[tag]["unknown"] = 0
        transfer[tag] = smooth(transfer[tag], factor)
    print(emission["NOUN"]["touchstone"])
    #get the UNKNOWN words' probilities
    #-------------------------make predications---------------------------
    predicts = []
    for sentence in test:
        temp = []
        dp_array = [[0 for i in range(len(record_tags))] for j in range(len(sentence))]
        record_array = [[0 for i in range(len(record_tags))] for j in range(len(sentence))]
        i = 0
        for j in range(len(record_tags)):
            dp_array[0][j] = get_initial(initial_count, record_tags[j]) + get_emission(emission,record_tags[j],sentence[i]) 
        i += 1
        while(i<len(sentence)):
            for j in range(len(record_tags)): #next col
                dp_array[i][j] = dp_array[i-1][0] + get_transfer(transfer, record_tags[0], record_tags[j]) + get_emission(emission,record_tags[j],sentence[i])
                record_array[i][j] = 0
                for k in range(len(record_tags)): #pre col
                    cmp = dp_array[i-1][k] + get_transfer(transfer, record_tags[k], record_tags[j]) + get_emission(emission,record_tags[j],sentence[i]) 
                    if(cmp > dp_array[i][j]):
                        dp_array[i][j] = cmp
                        record_array[i][j] = k
            i += 1
        #back track
        i = len(sentence) - 1
        dp_row = dp_array[i]
        spot = dp_row.index(max(dp_row))
        temp.append((sentence[i],record_tags[spot]))
        while(i >= 1):    
            temp.append((sentence[i-1],record_tags[record_array[i][spot]]))
            spot = record_array[i][spot]
            i -= 1
        temp = temp[::-1]
        predicts.append(temp)
    return predicts 