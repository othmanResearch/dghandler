from fhm.handi import InputHandler
from varimp import ReadData



if __name__ == "__main__":
    obj = InputHandler("protein")

    file = ReadData("../../BILIM/AGORA/data/tuberculosis_ClinPGX.tsv")
    print(obj)
