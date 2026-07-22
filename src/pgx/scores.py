import logging
import pandas as pd
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
