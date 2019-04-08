from __future__ import division, print_function
import numpy as np
import sys
import string
import operator
from collections import defaultdict
from string import punctuation
import math

def strip_punctuation_ss(s):
	return ''.join(c for c in s if c not in punctuation)

def find_Middle_Texts(start, end, line):
	Found_Word = ""
	if line.find(start):
		start_And_word = line[line.find(start):line.rfind(end)]
		Found_Word = start_And_word[len(start):]
		return Found_Word

def parse_file_ss(path):
	with open(path,'r') as data:
		ambiguous_Words_ss = ""
		num_s_Instances = 0
		for line in data.read().split('\n'):
			if line.startswith("<instance id="):
				num_s_Instances += 1
			start = '<head>'
			end = '</head>'
			if ambiguous_Words_ss == "":
				ambiguous_Words_ss = find_Middle_Texts(start, end, line)

	remainder_Instances_ss = num_s_Instances % 5
	num_s_Instances += (5 - remainder_Instances_ss)
	num_s_Instances_Per_Fold = num_s_Instances / 5
	return num_s_Instances, remainder_Instances_ss, num_s_Instances_Per_Fold, ambiguous_Words_ss

def parse_data_ss(path, num_s_Instances, remainder_Instances_ss, num_s_Instances_Per_Fold, ambiguous_Words_ss, num_Fold):
	test_Out_Files_Name = ambiguous_Words_ss + 'Test_Data.out'
	train_Out_Files_Name = ambiguous_Words_ss + 'Train_Data.out'
	Test_Data = open(test_Out_Files_Name, 'w')
	Train_Data = open(train_Out_Files_Name, 'w')

	Start_Instances = num_Fold * num_s_Instances_Per_Fold
	End_Instances = Start_Instances + num_s_Instances_Per_Fold

	with open(path,'r') as data:
		count_Instances_ss = 0
		for paragraph in data.read().split("\n\n"):
			if paragraph.startswith("<instance id="):
				count_Instances_ss += 1
				if count_Instances_ss >= Start_Instances and count_Instances_ss < End_Instances:
					Test_Data.write(paragraph)
					Test_Data.write("\n\n")
				else:
					Train_Data.write(paragraph)
					Train_Data.write("\n\n")
	Test_Data.close()
	Train_Data.close()
	return test_Out_Files_Name, train_Out_Files_Name

def parse_training_data(train_Out_Files_Name):
	sense_Dict = dict()
	num_Sense_Dict = dict()
	with open(train_Out_Files_Name, 'r') as data:
		for paragraph in data.read().split("\n\n"):
			for line in paragraph.split("\n"):
				if line.startswith("<answer"):
					start = "senseid=\""
					end = "\"/>"
					sense = find_Middle_Texts(start, end, line)
					if not sense in sense_Dict:
						sense_Dict[sense] = list()
						num_Sense_Dict[sense] = 0
					else:
						num_Sense_Dict[sense] += 1
				if line.startswith("<context>"):
					text = find_Middle_Texts("<context>", "</context>", paragraph)
					for word in text.split(" "):
						word = word.strip('\n')
						word = word.lower()
						word.strip()
						word = strip_punctuation_ss(word)
						if word != "":
							sense_Dict[sense].append(word)
	Unique_Sense_Dict = dict()
	Count_Unique_Sense_Dict = dict()
	for sense in sense_Dict:
		Count_Unique_Sense_Dict[sense] = 0
		Unique_Sense_Dict[sense] = list()
		for word in sense_Dict[sense]:
			if not word in Unique_Sense_Dict[sense]:
				Unique_Sense_Dict[sense].append(word)
		Count_Unique_Sense_Dict[sense] = len(Unique_Sense_Dict[sense])
	return sense_Dict, num_Sense_Dict, Unique_Sense_Dict, Count_Unique_Sense_Dict

def probabilities_Of_Sense_ss(num_Sense_Dict):
	prob_Sense_Dict = dict()
	Total_Num_Senses_ss = 0
	for sense in num_Sense_Dict:
		Total_Num_Senses_ss += num_Sense_Dict[sense]
	for sense in num_Sense_Dict:
		prob_Sense_Dict[sense] = num_Sense_Dict[sense] / Total_Num_Senses_ss
	return prob_Sense_Dict

def Extract_Test_Data(test_Out_Files_Name):
	words_And_Test_ID_Dict = dict()
	with open(test_Out_Files_Name, 'r') as data:
		for paragraph in data.read().split("\n\n"):
			for line in paragraph.split("\n"):
				if line.startswith("<instance"):
					start = "=\""
					end = "\" docsrc ="
					ID = find_Middle_Texts(start, end, line)
					if not ID in words_And_Test_ID_Dict:
						words_And_Test_ID_Dict[ID] = list()
				if line.startswith("<context>"):
					text = find_Middle_Texts("<context>", "</context>", paragraph)
					for word in text.split(" "):
						word = word.strip('\n')
						word = word.lower()
						word.strip()
						word = strip_punctuation_ss(word)
						if word != "":
							words_And_Test_ID_Dict[ID].append(word)
	return words_And_Test_ID_Dict

def Key_Of_Max_Value(scores_Dict):
	solved_Dict_ss = dict()
	Arg_Maxs = -999999999999
	label_Sense_ss = ""
	for ID in scores_Dict:
		for scores_List in scores_Dict[ID]:
			for sense in scores_List:
				score = scores_List[sense]
				if score > Arg_Maxs:
					Arg_Maxs = score
					label_Sense_ss = sense
		solved_Dict_ss[ID] = label_Sense_ss
	return solved_Dict_ss

def Naive_Bayes_Add_One_Smoothing(words_And_Test_ID_Dict, sense_Dict, num_Sense_Dict, Unique_Sense_Dict, prob_Sense_Dict, outFile):
	scores_Dict = dict()
	solved_Dict_ss = dict()
	outFile.write("Naive Bayes Add-One Smoothing")
	outFile.write("\n\n")
	for ID in words_And_Test_ID_Dict:
		for sense in sense_Dict:
			total = 0
			for word in words_And_Test_ID_Dict[ID]:
				numerator = sense_Dict[sense].count(word) + 1
				denominator = num_Sense_Dict[sense] + len(Unique_Sense_Dict[sense])
				total = math.log((numerator / denominator), 2)
			score = total + math.log(prob_Sense_Dict[sense], 2)
			if ID not in scores_Dict:
				scores_Dict[ID] = list()
			scores_Dict[ID].append({sense:score})
	solved_Dict_ss = Key_Of_Max_Value(scores_Dict)
	for ID in solved_Dict_ss:
		outFile.write(str(ID))
		outFile.write(" ")
		outFile.write(str(solved_Dict_ss[ID]))
		outFile.write("\n")
	return solved_Dict_ss

def Calculate_Accuracies(solved_Dict_ss, test_Out_Files_Name, outFile):
	solutionsDict = dict()
	with open(test_Out_Files_Name, 'r') as data:
		for paragraph in data.read().split("\n\n"):
			for line in paragraph.split("\n"):
				if line.startswith("<instance"):
					start = "=\""
					end = "\" docsrc ="
					ID = find_Middle_Texts(start, end, line)
				if line.startswith("<answer"):
					start = "senseid=\""
					end = "\"/>"
					sense = find_Middle_Texts(start, end, line)
					solutionsDict[ID] = sense
	Num_Correct = 0
	Num_Total = 0
	for key in solved_Dict_ss:
		if solved_Dict_ss[key] == solutionsDict[key]:
			Num_Correct += 1
		Num_Total += 1
	outFile.write("\n")
	outFile.write("Accuracy: ")
	outFile.write(str((Num_Correct/Num_Total)*100))
	outFile.write("%")
	outFile.write("\n\n")
	return Num_Correct/Num_Total
	print ("Accuracy: ",(str((Num_Correct/Num_Total)*100)))

def main():
	path = sys.argv[1]
	num_s_Instances, remainder_Instances_ss, num_s_Instances_Per_Fold, ambiguous_Words_ss = parse_file_ss(path)
	outputFileName = ambiguous_Words_ss+".wsd.out"
	outFile = open(outputFileName, 'w')
	Accuracies_Dict_ss = dict()
	for y in range(0, 5):
		outFile.write("Fold ")
		outFile.write(str(y))
		outFile.write("\n")

		test_Out_Files_Name, train_Out_Files_Name = parse_data_ss(path, num_s_Instances, remainder_Instances_ss, num_s_Instances_Per_Fold, ambiguous_Words_ss, y) #writes to new test and training files
		sense_Dict, num_Sense_Dict, Unique_Sense_Dict, Count_Unique_Sense_Dict = parse_training_data(train_Out_Files_Name) #parse training file
		prob_Sense_Dict = probabilities_Of_Sense_ss(num_Sense_Dict) 
		words_And_Test_ID_Dict = Extract_Test_Data(test_Out_Files_Name)

		solved_Dict_ss = Naive_Bayes_Add_One_Smoothing(words_And_Test_ID_Dict, sense_Dict, num_Sense_Dict, Unique_Sense_Dict, prob_Sense_Dict, outFile)
		accuracy = Calculate_Accuracies(solved_Dict_ss, test_Out_Files_Name, outFile)
		Accuracies_Dict_ss[y] = accuracy
		print ("Accuracy: ","Fold",str(y), accuracy * 100)
	Avg_Accuracy = 0

	for fold in Accuracies_Dict_ss:
		Avg_Accuracy += Accuracies_Dict_ss[fold]
	Avg_Accuracy = Avg_Accuracy/5

	outFile.write("\n")
	outFile.write("Average Accuracy: ")
	outFile.write(str(Avg_Accuracy*100))
	outFile.write("%\n")
	outFile.close()
	print ("Average Accuracy: ",(str(Avg_Accuracy*100)))

if __name__ == '__main__':
	main()
