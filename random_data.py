import random
import string
import numpy as np
import pandas as pd

# -----------------------------
# Parameters
# -----------------------------
N_ROWS = 100
MAX_GENE_REPETITIONS = 20
OUTPUT_FILE = "gene_allele_frequency_50_rows.csv"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Pool of gene symbols
gene_pool = [
    "ABCB1", "ABCC1", "ABCG2", "ACE", "ADH1B", "ADRB2", "APOE",
    "ATM", "BCHE", "CYP1A1", "CYP1A2", "CYP2B6", "CYP2C19",
    "CYP2C9", "CYP2D6", "CYP2E1", "CYP3A4", "CYP3A5",
    "DPYD", "EGFR", "ESR1", "F5", "F7", "G6PD",
    "GSTP1", "HLA-B", "IFNL3", "IL6", "JAK2", "KCNJ11",
    "KRAS", "MTHFR", "NAT1", "NAT2", "NQO1", "PAH",
    "PIK3CA", "RYR1", "SLCO1B1", "SLC22A1", "SLC22A2",
    "SLC6A4", "TP53", "UGT1A1", "VKORC1", "VDR",
    "XPC", "XRCC1", "CYP4F2", "NUDT15"
]


# -----------------------------
# Generate random IDs
# -----------------------------
def random_id(prefix="VAR", length=8):
    return prefix + "_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=length)
    )


# -----------------------------
# Generate allele frequency
# -----------------------------
def random_allele_frequency():
    r = random.random()

    # Missing value
    if r < 0.10:
        return np.nan

    # "-"
    elif r < 0.20:
        return "-"

    # Scientific notation
    elif r < 0.40:
        value = 10 ** random.uniform(-6, -1)
        return f"{value:.2e}"

    # Regular float
    else:
        return round(random.uniform(0, 1), 6)


# -----------------------------
# Sample genes with replacement
# while ensuring no gene appears
# more than MAX_GENE_REPETITIONS
# -----------------------------
counts = {gene: 0 for gene in gene_pool}
genes = []

while len(genes) < N_ROWS:
    gene = random.choice(gene_pool)

    if counts[gene] < MAX_GENE_REPETITIONS:
        genes.append(gene)
        counts[gene] += 1


# -----------------------------
# Create dataframe
# -----------------------------
df = pd.DataFrame({
    "id": [random_id() for _ in range(N_ROWS)],
    "genes": genes,
    "allele_freq": [random_allele_frequency() for _ in range(N_ROWS)]
})

# Save CSV
df.to_csv(OUTPUT_FILE, index=False)

print(df.head())
print(f"\nCSV written to: {OUTPUT_FILE}")

print("\nGene counts:")
print(df["genes"].value_counts())
