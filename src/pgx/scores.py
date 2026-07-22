import logging
import pandas as pd
import numpy as np
from .read_data import ReadData
logger = logging.getLogger(__name__)


class FunctionalVariantScorer:
    """
    Compute CAP (per-gene) and DRP (per-drug) functional-variant risk
    scores from a ReadData object's standardized variant-annotation table.

    Parameters
    ----------
    reader : ReadData
        An already-constructed ReadData instance. Its
        ``get_allele_frequency()`` output supplies the standardized
        "var_id", "symbol", "AF", "drug_id" columns used by this class.
    mapping, gene_symbols, strict :
        Forwarded as-is to ``reader.get_data_for_scoring()``.
    """

    def __init__(self, reader: ReadData, mapping=None, gene_symbols=None, strict=True):
        if not isinstance(reader, ReadData):
            raise TypeError(
                f"'reader' must be a ReadData instance, got {type(reader).__name__}."
            )

        self.reader = reader
        self.data = reader.get_data_for_scoring( mapping=mapping, gene_symbols=gene_symbols, strict=strict)

        self._cap_scores: pd.Series | None = None
        self._drp_scores: pd.Series | None = None

    def compute_CAP(self) -> pd.Series:
        """
        Compute the Cumulative Allele Probability (CAP) score per gene.

        CAP(g) = 1 - prod_{a in A_g} (1 - AF(a))**2

        where A_g is the set of distinct functional variants annotated
        to gene g. No gene-length normalization is applied here (this
        corresponds to "Pipeline A" / the unweighted CAP as defined in
        Scharfe et al. 2017).

        Returns
        -------
        pandas.Series
            Indexed by gene symbol ("symbol"), values are CAP scores
            in [0, 1]. Name of the series is "CAP".
        """
        # A variant may be linked to several drugs (appearing on multiple
        # rows with the same var_id/symbol/AF). It must only contribute
        # once to its gene's CAP score, regardless of how many drugs
        # reference it.
        distinct_variants = self.data.drop_duplicates(subset=["var_id", "symbol"]).copy()

        # CAP(g) = 1 - prod(1 - AF)^2
        #        = 1 - exp( sum( 2 * log(1 - AF) ) )
        # Computed in log-space and aggregated with a plain groupby-sum, numerical underflow when a gene has many variants.
        distinct_variants["_log_term"] = 2.0 * np.log1p(-distinct_variants["AF"])

        log_sum_per_gene = distinct_variants.groupby("symbol")["_log_term"].sum()
        cap_scores = (1.0 - np.exp(log_sum_per_gene)).rename("CAP")

        self._cap_scores = cap_scores
        return cap_scores

    def compute_DRP(self) -> pd.Series:
        """
        Compute the Drug Risk Probability (DRP) score per drug.

        DRP(d) = 1 - prod_{g in G_d} CAP(g)

        where G_d is the set of distinct target genes for drug d. Requires
        ``compute_CAP`` to have been run (called automatically if not
        already done).

        Returns
        -------
        pandas.Series
            Indexed by drug_id, values are DRP scores in [0, 1]. Name
            of the series is "DRP".
        """
        if self._cap_scores is None:
            self.compute_CAP()

        # Each distinct gene must contribute to a drug's DRP exactly once,
        # even if several of the gene's variants are linked to that same
        # drug (i.e., appear as separate rows in self.data).
        gene_drug_pairs = self.data[["symbol", "drug_id"]].drop_duplicates().copy()

        gene_drug_pairs = gene_drug_pairs.merge(
            self._cap_scores, left_on="symbol", right_index=True, how="left"
        )

        unmatched = gene_drug_pairs["CAP"].isna()
        if unmatched.any():
            logger.warning(
                "%d (gene, drug) pairs have no matching CAP score and "
                "will be excluded from DRP computation.",
                int(unmatched.sum()),
            )
            gene_drug_pairs = gene_drug_pairs.loc[~unmatched]

        # DRP(d) = 1 - prod(CAP(g))
        #        = 1 - exp( sum( log(CAP(g)) ) )
        # Same log-space aggregation trick as compute_CAP, avoiding a
        # nested function and numerical underflow for drugs with many
        # target genes.
        #
        # A gene with CAP(g) == 0 (no functional variants observed) is a
        # valid, expected input: log(0) = -inf is mathematically correct
        # here (exp(-inf) = 0 in the product, correctly forcing DRP = 1
        # for that drug). We suppress the RuntimeWarning numpy would
        # otherwise raise for log(0), since this case is handled
        # correctly rather than being a computation error.
        if (gene_drug_pairs["CAP"] == 0).any():
            n_zero_cap = int((gene_drug_pairs["CAP"] == 0).sum())
            logger.info(
                "%d (gene, drug) pairs have CAP(g) == 0 (gene has no "
                "functional variants); this correctly forces DRP = 1 for "
                "the affected drug(s).",
                n_zero_cap,
            )

        with np.errstate(divide="ignore"):
            gene_drug_pairs["_log_cap"] = np.log(gene_drug_pairs["CAP"])

        log_sum_per_drug = gene_drug_pairs.groupby("drug_id")["_log_cap"].sum()
        drp_scores = (1.0 - np.exp(log_sum_per_drug)).rename("DRP")

        self._drp_scores = drp_scores
        return drp_scores
