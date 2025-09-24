import os
import sys
from toolkit_mlp.describe_utils import describe
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main():
    try:
        assert len(sys.argv) == 2, "You must provide the dataset file path"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        describe(sys.argv[1])

    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
