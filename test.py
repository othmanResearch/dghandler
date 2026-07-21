from fhm.handi import InputHandler
from varimp import ReadData



if __name__ == "__main__":
    obj = InputHandler("protein")

    file = ReadData("./gene_allele_frequency_50_rows.csv")
    
    print()
    file.get_allele_frequency(mapping = {"AF":"allele_freq", "symbol":"genes", "var_id":"id"}, strict=False)
