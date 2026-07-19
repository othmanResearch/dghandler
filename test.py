from fhm.handi import InputHandler


def test_single_string():
    """
    Test initialization with a single string.
    """

    obj = InputHandler("protein.pdb")

if __name__ == "__main__":
    obj = InputHandler("protein")
    print(obj)
