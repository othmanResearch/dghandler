import logging
import pandas as pd
import numpy as np
from .read_data import ReadData
logger = logging.getLogger(__name__)

_EVIDENCE_LEVEL_TO_SCORE = {
        "1A": 5,
        "1B": 4,
        "2A": 3,
        "2B": 2,
        "3": 1,
        "4": 0,
    }

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
        self._wcap_scores: pd.Series | None = None
        self._wdrp_scores: pd.Series | None = None
        self._des_dominant_scores: pd.Series | None = None
        self._des_recessive_scores: pd.Series | None = None
        self._wdes_dominant_scores: pd.Series | None = None
        self._wdes_recessive_scores: pd.Series | None = None

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

    def compute_wCAP(self, gene_coords_path: str) -> pd.Series:
        """
        Compute the gene-length-weighted CAP (wCAP) score per gene.

        Building on the cumulative hazard framework (Aalen, 1978), CAP is
        converted to a hazard, normalized by gene length in kb, and
        transformed back to the probability scale:

            I(g)         = -ln(CAP(g))
            I(g, per_kb) = I(g) / (length_kb(g))
            wCAP(g)      = 1 - exp(-I(g, per_kb))

        Requires ``compute_CAP`` to have been run (called automatically
        if not already done). Gene lengths are derived from a mandatory
        coordinate file supplied by the caller (see
        ``_load_gene_lengths_kb`` for the expected file format).

        Parameters
        ----------
        gene_coords_path : str
            Path to a CSV, TSV, or Excel (.xlsx/.xls) file containing at
            minimum the columns "symbol", "start", and "end". Gene length
            is computed as (max(end) - min(start)) / 1000 for each symbol,
            in case a gene has multiple rows (e.g., multiple transcripts).

        Returns
        -------
        pandas.Series
            Indexed by gene symbol ("symbol"), values are wCAP scores
            in [0, 1]. Name of the series is "wCAP".

        Raises
        ------
        ValueError
            If the coordinate file is missing "symbol", "start", or "end",
            or if any gene present in the CAP scores is absent from the
            coordinate file.
        """
        if self._cap_scores is None:
            self.compute_CAP()

        gene_lengths_kb = self._load_gene_lengths_kb(gene_coords_path)

        missing_genes = sorted(set(self._cap_scores.index) - set(gene_lengths_kb.index))
        if missing_genes:
            raise ValueError(
                "The following gene(s) from the CAP scores were not found "
                f"in the coordinate file '{gene_coords_path}': "
                f"{', '.join(missing_genes)}"
            )

        aligned_lengths_kb = gene_lengths_kb.reindex(self._cap_scores.index)

        # I(g) = -ln(CAP(g)); CAP(g) == 0 correctly yields I(g) = +inf,
        # which in turn yields wCAP(g) = 1 (a gene with no functional
        # variants contributes no additional risk, so this edge case is
        # handled naturally by the math rather than needing a special case).
        with np.errstate(divide="ignore"):
            hazard = -np.log(self._cap_scores)

        hazard_per_kb = hazard / aligned_lengths_kb
        wcap_scores = (1.0 - np.exp(-hazard_per_kb)).rename("wCAP")

        self._wcap_scores = wcap_scores
        return wcap_scores

    @staticmethod
    def _load_gene_lengths_kb(gene_coords_path: str) -> pd.Series:
        """
        Load a gene coordinate file and compute gene length in kb per symbol.

        Accepts .csv, .tsv, or Excel (.xlsx/.xls) files. The file must
        contain "symbol", "start", and "end" columns (case-sensitive).

        Returns
        -------
        pandas.Series
            Indexed by "symbol", values are gene lengths in kb.

        Raises
        ------
        ValueError
            If any of "symbol", "start", or "end" is missing from the
            file, or if the file extension is unsupported.
        """
        suffix = gene_coords_path.rsplit(".", 1)[-1].lower()
        if suffix in ("xlsx", "xls"):
            coords = pd.read_excel(gene_coords_path)
        elif suffix == "tsv":
            coords = pd.read_csv(gene_coords_path, sep="\t")
        elif suffix == "csv":
            coords = pd.read_csv(gene_coords_path)
        else:
            raise ValueError(
                f"Unsupported file extension '.{suffix}' for gene "
                "coordinate file. Expected .csv, .tsv, .xlsx, or .xls."
            )

        required_cols = {"symbol", "start", "end"}
        missing_cols = required_cols - set(coords.columns)
        if missing_cols:
            raise ValueError(
                "Gene coordinate file is missing required column(s): "
                f"{', '.join(sorted(missing_cols))}. Expected columns: "
                f"{', '.join(sorted(required_cols))}."
            )

        # A gene may span multiple rows (e.g., multiple transcripts);
        # take the outer span (min start to max end) as the gene length.
        span = coords.groupby("symbol").agg(_min_start=("start", "min"), _max_end=("end", "max"))
        gene_lengths_kb = ((span["_max_end"] - span["_min_start"]).abs() / 1000.0).rename("length_kb")

        return gene_lengths_kb


    def compute_wDRP(self, gene_coords_path: str | None = None) -> pd.Series:
        """
        Compute the weighted Drug Risk Probability (wDRP) score per drug.

        wDRP(d) = 1 - prod_{g in G_d} wCAP(g)

        where G_d is the set of distinct target genes for drug d. Requires
        ``compute_wCAP`` to have been run; if it hasn't, ``gene_coords_path``
        must be supplied so it can be computed here.

        Parameters
        ----------
        gene_coords_path : str, optional
            Path to the gene coordinate file (CSV/TSV/Excel with "symbol",
            "start", "end" columns), forwarded to ``compute_wCAP`` if wCAP
            scores haven't already been computed. Not required if
            ``compute_wCAP`` was already called on this instance.

        Returns
        -------
        pandas.Series
            Indexed by drug_id, values are wDRP scores in [0, 1]. Name
            of the series is "wDRP".

        Raises
        ------
        ValueError
            If wCAP scores are not yet available and ``gene_coords_path``
            is not supplied.
        """
        if self._wcap_scores is None:
            if gene_coords_path is None:
                raise ValueError(
                    "wCAP scores have not been computed yet. Either call "
                    "compute_wCAP(gene_coords_path) first, or pass "
                    "gene_coords_path to compute_wDRP() so it can be "
                    "computed automatically."
                )
            self.compute_wCAP(gene_coords_path)

        # Each distinct gene must contribute to a drug's wDRP exactly once,
        # even if several of the gene's variants are linked to that same
        # drug (i.e., appear as separate rows in self.data).
        gene_drug_pairs = self.data[["symbol", "drug_id"]].drop_duplicates().copy()

        gene_drug_pairs = gene_drug_pairs.merge(
            self._wcap_scores, left_on="symbol", right_index=True, how="left"
        )

        unmatched = gene_drug_pairs["wCAP"].isna()
        if unmatched.any():
            logger.warning(
                "%d (gene, drug) pairs have no matching wCAP score and "
                "will be excluded from wDRP computation.",
                int(unmatched.sum()),
            )
            gene_drug_pairs = gene_drug_pairs.loc[~unmatched]

        # wDRP(d) = 1 - prod(wCAP(g))
        #         = 1 - exp( sum( log(wCAP(g)) ) )
        # Same log-space aggregation as compute_DRP. A gene with
        # wCAP(g) == 0 is a valid input (no functional variants observed,
        # normalized): log(0) = -inf correctly forces wDRP = 1 for that
        # drug, so the RuntimeWarning is suppressed rather than treated
        # as an error.
        if (gene_drug_pairs["wCAP"] == 0).any():
            n_zero_wcap = int((gene_drug_pairs["wCAP"] == 0).sum())
            logger.info(
                "%d (gene, drug) pairs have wCAP(g) == 0 (gene has no "
                "functional variants); this correctly forces wDRP = 1 for "
                "the affected drug(s).",
                n_zero_wcap,
            )

        with np.errstate(divide="ignore"):
            gene_drug_pairs["_log_wcap"] = np.log(gene_drug_pairs["wCAP"])

        log_sum_per_drug = gene_drug_pairs.groupby("drug_id")["_log_wcap"].sum()
        wdrp_scores = (1.0 - np.exp(log_sum_per_drug)).rename("wDRP")

        self._wdrp_scores = wdrp_scores
        return wdrp_scores

    def compute_DES_dominant(self) -> pd.Series:
        """
        Compute the Drug Effect Score (DES) per drug under the dominant
        model (carrying at least one risk allele is sufficient to alter
        drug response).

            DES_dominant(d) = prod_{a in D_d} (1 - AF(a))**2

        where D_d is the set of distinct PGx SNPs affecting response to
        drug d. Unlike CAP/DRP, DES is the product itself (representing
        the probability an individual carries no risk alleles for drug d),
        not its complement.

        Returns
        -------
        pandas.Series
            Indexed by drug_id, values are DES scores in [0, 1]. Name
            of the series is "DES_dominant".
        """
        # A variant may appear on multiple rows (e.g., once per gene it's
        # annotated to), but must contribute only once per drug.
        distinct_variants = self.data.drop_duplicates(subset=["var_id", "drug_id"]).copy()

        # DES(d) = prod(1 - AF)^2 = exp( sum( 2 * log(1 - AF) ) )
        distinct_variants["_log_term"] = 2.0 * np.log1p(-distinct_variants["AF"])

        log_sum_per_drug = distinct_variants.groupby("drug_id")["_log_term"].sum()
        des_dominant_scores = np.exp(log_sum_per_drug).rename("DES_dominant")

        self._des_dominant_scores = des_dominant_scores
        return des_dominant_scores

    def compute_DES_recessive(self) -> pd.Series:
        """
        Compute the Drug Effect Score (DES) per drug under the recessive
        model (two copies of a risk allele are required to alter drug
        response).

            DES_recessive(d) = prod_{a in D_d} (1 - AF(a)**2)

        where D_d is the set of distinct PGx SNPs affecting response to
        drug d.

        Returns
        -------
        pandas.Series
            Indexed by drug_id, values are DES scores in [0, 1]. Name
            of the series is "DES_recessive".
        """
        distinct_variants = self.data.drop_duplicates(subset=["var_id", "drug_id"]).copy()

        # DES(d) = prod(1 - AF^2) = exp( sum( log(1 - AF^2) ) )
        distinct_variants["_log_term"] = np.log1p(-distinct_variants["AF"] ** 2)

        log_sum_per_drug = distinct_variants.groupby("drug_id")["_log_term"].sum()
        des_recessive_scores = np.exp(log_sum_per_drug).rename("DES_recessive")

        self._des_recessive_scores = des_recessive_scores
        return des_recessive_scores

    def compute_wDES_dominant(self, evidence_path: str, alpha: float = 1.5, beta: float = 2.5) -> pd.Series:
        """
        Compute the weighted Drug Effect Score (wDES) per drug under the
        dominant model, down-weighting variants by evidence-level confidence.

            w(a)   = 1 / (1 + exp{-alpha * (S(a) - beta)})
            wDES_dominant(d) = prod_{a in D_d} (1 - AF(a))**(2 * w(a))

        Parameters
        ----------
        evidence_path : str
            Path to a CSV, TSV, or Excel file with at least the columns
            "var_id", "drug_id", "evidence_level". Values in
            "evidence_level" must be one of {"1A","1B","2A","2B","3","4"}.
        alpha, beta : float
            Sigmoid steepness and inflection-point parameters (defaults
            1.5 and 2.5, per the paper's sensitivity analysis).

        Returns
        -------
        pandas.Series
            Indexed by drug_id, values are wDES scores in [0, 1]. Name
            of the series is "wDES_dominant".

        Raises
        ------
        ValueError
            If the evidence file is missing required columns, contains
            invalid evidence_level values, or is missing an entry for
            any (var_id, drug_id) pair present in self.data.
        """
        weighted = self._prepare_wdes_variants(evidence_path, alpha, beta)

        # wDES_dominant(d) = prod((1-AF)^(2w)) = exp( sum( 2*w*log(1-AF) ) )
        weighted["_log_term"] = 2.0 * weighted["_weight"] * np.log1p(-weighted["AF"])
        log_sum_per_drug = weighted.groupby("drug_id")["_log_term"].sum()
        wdes_dominant_scores = np.exp(log_sum_per_drug).rename("wDES_dominant")

        self._wdes_dominant_scores = wdes_dominant_scores
        return wdes_dominant_scores

    def compute_wDES_recessive(
        self, evidence_path: str, alpha: float = 1.5, beta: float = 2.5
    ) -> pd.Series:
        """
        Compute the weighted Drug Effect Score (wDES) per drug under the
        recessive model, down-weighting variants by evidence-level confidence.

            w(a)   = 1 / (1 + exp{-alpha * (S(a) - beta)})
            wDES_recessive(d) = prod_{a in D_d} (1 - AF(a)**2)**w(a)

        Parameters, Returns, and Raises: see ``compute_wDES_dominant``.
        """
        weighted = self._prepare_wdes_variants(evidence_path, alpha, beta)

        # wDES_recessive(d) = prod((1-AF^2)^w) = exp( sum( w*log(1-AF^2) ) )
        weighted["_log_term"] = weighted["_weight"] * np.log1p(-weighted["AF"] ** 2)
        log_sum_per_drug = weighted.groupby("drug_id")["_log_term"].sum()
        wdes_recessive_scores = np.exp(log_sum_per_drug).rename("wDES_recessive")

        self._wdes_recessive_scores = wdes_recessive_scores
        return wdes_recessive_scores

    def _prepare_wdes_variants(
        self, evidence_path: str, alpha: float, beta: float
    ) -> pd.DataFrame:
        """
        Merge self.data with per-(var_id, drug_id) evidence levels and
        compute the sigmoid weight w(a) for each distinct variant-drug pair.

        Shared setup logic for compute_wDES_dominant and
        compute_wDES_recessive.
        """
        # A variant must contribute only once per drug (mirrors compute_DES_*).
        distinct_variants = self.data.drop_duplicates(subset=["var_id", "drug_id"]).copy()

        evidence = self._load_evidence_levels(evidence_path)

        merged = distinct_variants.merge(evidence, on=["var_id", "drug_id"], how="left")

        missing_mask = merged["evidence_level"].isna()
        if missing_mask.any():
            missing_pairs = (
                merged.loc[missing_mask, ["var_id", "drug_id"]]
                .drop_duplicates()
                .itertuples(index=False)
            )
            pairs_str = ", ".join(f"({v}, {d})" for v, d in missing_pairs)
            raise ValueError(
                "The following (var_id, drug_id) pair(s) present in the "
                f"data have no matching evidence_level in "
                f"'{evidence_path}': {pairs_str}"
            )

        merged["_score"] = merged["evidence_level"].map(self._EVIDENCE_LEVEL_TO_SCORE)
        merged["_weight"] = 1.0 / (1.0 + np.exp(-alpha * (merged["_score"] - beta)))

        return merged

    @staticmethod
    def _load_evidence_levels(evidence_path: str) -> pd.DataFrame:
        """
        Load and validate a variant-drug evidence-level file.

        Accepts .csv, .tsv, or Excel (.xlsx/.xls) files. Must contain
        "var_id", "drug_id", and "evidence_level" columns. Values in
        "evidence_level" must be one of {"1A","1B","2A","2B","3","4"}.

        Returns
        -------
        pandas.DataFrame
            Deduplicated ["var_id", "drug_id", "evidence_level"] table.

        Raises
        ------
        ValueError
            If required columns are missing, the file extension is
            unsupported, or invalid evidence_level values are present.
        """
        suffix = evidence_path.rsplit(".", 1)[-1].lower()
        if suffix in ("xlsx", "xls"):
            evidence = pd.read_excel(evidence_path)
        elif suffix == "tsv":
            evidence = pd.read_csv(evidence_path, sep="\t")
        elif suffix == "csv":
            evidence = pd.read_csv(evidence_path)
        else:
            raise ValueError(
                f"Unsupported file extension '.{suffix}' for evidence "
                "level file. Expected .csv, .tsv, .xlsx, or .xls."
            )

        required_cols = {"var_id", "drug_id", "evidence_level"}
        missing_cols = required_cols - set(evidence.columns)
        if missing_cols:
            raise ValueError(
                "Evidence level file is missing required column(s): "
                f"{', '.join(sorted(missing_cols))}. Expected columns: "
                f"{', '.join(sorted(required_cols))}."
            )

        evidence = evidence[["var_id", "drug_id", "evidence_level"]].drop_duplicates().copy()
        evidence["evidence_level"] = evidence["evidence_level"].astype(str).str.strip()

        allowed_levels = set(FunctionalVariantScorer._EVIDENCE_LEVEL_TO_SCORE.keys())
        invalid_mask = ~evidence["evidence_level"].isin(allowed_levels)
        if invalid_mask.any():
            invalid_values = sorted(evidence.loc[invalid_mask, "evidence_level"].unique())
            raise ValueError(
                f"Invalid evidence_level value(s) found: {invalid_values}. "
                f"Allowed values are: {sorted(allowed_levels)}."
            )

        return evidence
