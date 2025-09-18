import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from toolkit_mlp.utils import draw_histogram


def main():
    try:
        assert len(sys.argv) == 2, "You must provide the dataset file path"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        draw_histogram(sys.argv[1])

    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)

if __name__ == "__main__":
    main()
