import sys
import neuralNet

def main(argv):
	if argv[0] == '-t':
		neuralNet.trainModel(argv[1], argv[2])
	elif argv[0] == '-e':
		neuralNet.evaluateModel(argv[1], argv[2])
	elif argv[0] == '-v':
		neuralNet.visualizeModelResults(argv[1], argv[2])
		

main(sys.argv[1:])
