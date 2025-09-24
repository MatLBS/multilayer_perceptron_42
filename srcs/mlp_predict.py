import sys
import os
import pickle

def predict_model(file: str):
    with open(file, 'rb') as file:  
        model = pickle.load(file)

    # evaluate model 
    weights = model.weights
    
    print(weights)

    # # check results
    # print(classification_report(y_test, y_predict)) 

def main():
    try:
        assert len(sys.argv) == 2, "You must provide the dataset file path"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        predict_model(sys.argv[1])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
