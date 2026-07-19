from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder


@dataclass
class Structure:

    file: Path
    format: str | None = None

    def __post_init__(self):

        self.file = Path(self.file)

        if not self.file.exists():
            raise FileNotFoundError(
                f"Structure file does not exist: {self.file}"
            )

        if self.format is None:
            self.format = self._detect_format()

        self._structure = None


    def _detect_format(self):

        suffix = self.file.suffix.lower()

        if suffix in [".pdb"]:
            return "pdb"

        elif suffix in [".cif", ".mmcif"]:
            return "mmcif"

        else:
            raise ValueError(
                f"Unsupported structure format: {suffix}"
            )


    def load(self):

        if self._structure is None:

            if self.format == "pdb":

                parser = PDBParser(
                    QUIET=True
                )

            elif self.format == "mmcif":

                parser = MMCIFParser(
                    QUIET=True
                )

            self._structure = parser.get_structure(
                self.file.stem,
                self.file
            )

        return self._structure


    def get_sequences(self):

        structure = self.load()

        pp_builder = PPBuilder()

        sequences = {}

        model = structure[0]

        for chain in model:

            peptides = pp_builder.build_peptides(chain)

            sequence = ""

            for peptide in peptides:
                sequence += str(
                    peptide.get_sequence()
                )

            sequences[chain.id] = sequence

        return sequences
