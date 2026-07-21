from pathlib import Path
import pandas as pd
import logging

class ReadData:
    """Read tabular data from CSV, TSV, and Excel files."""

    SUPPORTED_FORMATS = { ".csv", ".tsv", ".xls", ".xlsx"}

    
    def __init__(self, file_path):
        """
        Initialize the ReadData object.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the input data file.
        """
        self.file_path = Path(file_path)

        self._validate_file()
        self.data = self._read_file()

    def _validate_file(self):
        """Validate that the file exists and has a supported format."""

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. "
                f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )

    
    def _read_file(self):
        """Read the input file and return a pandas DataFrame."""

        suffix = self.file_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(self.file_path)

        elif suffix == ".tsv":
            return pd.read_csv(self.file_path, sep="\t")

        elif suffix in {".xls", ".xlsx"}:
            return pd.read_excel(self.file_path)

        # This should never be reached because of validation
        raise ValueError(f"Unsupported file format: {suffix}")

    def get_data(self):
        """Return the loaded data as a pandas DataFrame."""
        return self.data


    def get_allele_frequency(self, mapping=None, gene_symbols=None, strict=True ):
        """
        Extract and validate allele-frequency and gene-symbol data.

        Parameters
        ----------
        mapping : dict, optional
            Mapping between the standard column names and the actual
            column names in the input dataset.

            Expected keys are:

                {
                    "AF": "actual_allele_frequency_column",
                    "symbol": "actual_gene_symbol_column",
                }

            If None, the default columns "AF" and "symbol" are used.

        gene_symbols : list of str, optional
            List of gene symbols to retain. If None, all genes are retained.

        strict : bool, default=True
            Validation mode.

            If True, an error is raised if missing or invalid values are
            detected in either the AF or symbol column.

            If False, values that can be converted are retained and rows
            containing values that cannot be converted or are missing are
            removed. A summary of the removed rows is logged.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing two standardized columns:

                - "symbol"
                - "AF"

        Raises
        ------
        ValueError
            If invalid or missing values are found in strict mode.

        KeyError
            If the required input columns do not exist.
        """

        # ---------------------------------------------------------
        # 1. Define the default mapping
        # ---------------------------------------------------------

        if mapping is None:
            mapping = {
                "AF": "AF",
                "symbol": "symbol",
            }

        # ---------------------------------------------------------
        # 2. Validate the mapping
        # ---------------------------------------------------------

        required_keys = {"AF", "symbol"}

        if set(mapping.keys()) != required_keys:
            raise ValueError(
                "The mapping must contain exactly the following keys: "
                "'AF' and 'symbol'."
            )

        af_column = mapping["AF"]
        symbol_column = mapping["symbol"]

        # ---------------------------------------------------------
        # 3. Check that the columns exist
        # ---------------------------------------------------------

        missing_columns = [
            column
            for column in [af_column, symbol_column]
            if column not in self.data.columns
        ]

        if missing_columns:
            raise KeyError(
                f"The following required columns were not found: "
                f"{missing_columns}. "
                f"Available columns are: "
                f"{list(self.data.columns)}"
                f"Redefine the columns' names in the input file or use the mapping option"
            )

        # ---------------------------------------------------------
        # 4. Select the required columns
        # ---------------------------------------------------------

        result = self.data[
            [symbol_column, af_column]
        ].copy()

        # Standardize the column names
        result = result.rename(
            columns={
                symbol_column: "symbol",
                af_column: "AF",
            }
        )

        # ---------------------------------------------------------
        # 5. Validate the AF column
        # ---------------------------------------------------------

        # Count missing AF values before conversion
        missing_af = result["AF"].isna()

        # Attempt to convert all AF values to numeric
        converted_af = pd.to_numeric(
            result["AF"],
            errors="coerce",
        )

        # Values that were not missing but could not be converted
        invalid_af = (
            converted_af.isna()
            & ~missing_af
        )

        n_missing_af = missing_af.sum()
        n_invalid_af = invalid_af.sum()

        # ---------------------------------------------------------
        # 6. Validate the symbol column
        # ---------------------------------------------------------

        missing_symbol = result["symbol"].isna()

        # Convert values to strings where possible
        converted_symbol = result["symbol"].astype("string")

        # A value is considered invalid if it is not a string
        invalid_symbol = (
            ~result["symbol"].map(lambda x: isinstance(x, str))
            & ~missing_symbol
        )

        n_missing_symbol = missing_symbol.sum()
        n_invalid_symbol = invalid_symbol.sum()

        # ---------------------------------------------------------
        # 7. Strict validation
        # ---------------------------------------------------------

        if strict:

            problems = []

            if n_missing_af > 0:
                problems.append(
                    f"{n_missing_af} missing values in the AF column"
                )

            if n_invalid_af > 0:
                problems.append(
                    f"{n_invalid_af} AF values could not be converted "
                    "to float"
                )

            if n_missing_symbol > 0:
                problems.append(
                    f"{n_missing_symbol} missing values in the "
                    "symbol column"
                )

            if n_invalid_symbol > 0:
                problems.append(
                    f"{n_invalid_symbol} symbol values are not strings"
                )

            if problems:
                message = (
                    "Data validation failed:\n- "
                    + "\n- ".join(problems)
                    + "\n\n"
                    "If you are sure that problematic rows can be "
                    "removed, rerun the method with strict=False."
                )

                raise ValueError(message)

        # ---------------------------------------------------------
        # 8. Lenient validation
        # ---------------------------------------------------------

        else:

            # Log a summary of problematic values
            logging.warning(
                "Data validation summary:"
            )

            logging.warning(
                "AF values that could not be converted: %d",
                n_invalid_af,
            )

            logging.warning(
                "Missing AF values: %d",
                n_missing_af,
            )

            logging.warning(
                "Symbol values that are not strings: %d",
                n_invalid_symbol,
            )

            logging.warning(
                "Missing symbol values: %d",
                n_missing_symbol,
            )

            # Keep only rows with valid AF values
            valid_af = converted_af.notna()

            # Keep only rows with valid string symbols
            valid_symbol = (
                result["symbol"].map(
                    lambda x: isinstance(x, str)
                )
            )

            # Apply the validation
            valid_rows = valid_af & valid_symbol

            result = result.loc[valid_rows].copy()

            # Use the converted AF values
            result["AF"] = converted_af.loc[
                valid_rows
            ].astype(float)

            # Standardize symbol values as strings
            result["symbol"] = converted_symbol.loc[
                valid_rows
            ]

        # ---------------------------------------------------------
        # 9. Apply gene-symbol filtering
        # ---------------------------------------------------------

        if gene_symbols is not None:
            result = result[
                result["symbol"].isin(gene_symbols)
            ]
        print(result)
        return result


