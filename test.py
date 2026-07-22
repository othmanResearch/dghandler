from fhm.handi import InputHandler
from pgx import ReadData, FunctionalVariantScorer

if __name__ == "__main__":
    obj = InputHandler("protein")
    file = ReadData("./gene_allele_frequency_50_rows.csv")

    print()
    #file.get_allele_frequency(
    #    mapping={"AF": "allele_freq", "symbol": "genes", "var_id": "id", "drug_id": "drug_id"},
    #    strict=False,
    #)

    scorer = FunctionalVariantScorer(
        file,
        mapping={"AF": "allele_freq", "symbol": "genes", "var_id": "id", "drug_id": "drug_id"},
        strict=False,
    )

