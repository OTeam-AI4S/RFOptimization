"""Adapter from RFO's cycle contract to OFoldX's ProteinMPNN/LigandMPNN pipelines."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from functools import lru_cache
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBIOWarning
from Bio.SeqUtils import seq1

MODEL_IDS = {
    "protein_mpnn": "oteam/proteinmpnn",
    "ligand_mpnn": "oteam/ligandmpnn",
}
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def settings():
    """Resolve the OFoldX model selection without loading model weights."""
    model_type = os.environ.get("MPNN_MODEL_TYPE", "protein_mpnn")
    if model_type not in MODEL_IDS:
        raise ValueError(f"MPNN_MODEL_TYPE must be one of {tuple(MODEL_IDS)}")
    model = os.environ.get("MPNN_MODEL") or MODEL_IDS[model_type]
    try:
        import ofoldx
    except ImportError as exc:
        raise ImportError(
            "RFO's MPNN adapter now uses OFoldX. Install OFoldX in the RFO "
            "Python environment and configure MPNN_MODEL_TYPE/MPNN_MODEL; "
            "see docs/installation.md."
        ) from exc
    if not callable(getattr(ofoldx, "pipeline", None)):
        raise ImportError("The installed OFoldX package does not expose ofoldx.pipeline.")
    return model_type, model


@lru_cache(maxsize=4)
def _load_pipeline(model_type: str, model: str):
    """Load and reuse an OFoldX design pipeline across cycles."""
    import ofoldx
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = ofoldx.pipeline("design", model=model, device=device)
    expected = {
        "protein_mpnn": "ProteinMpnnPipeline",
        "ligand_mpnn": "LigandMpnnPipeline",
    }[model_type]
    if type(selected).__name__ != expected:
        raise TypeError(
            f"MPNN_MODEL_TYPE={model_type!r} requires an OFoldX {expected}; "
            f"model {model!r} loaded {type(selected).__name__}."
        )
    return selected


def chain_residues(structure, chain_id):
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Designed chain {chain_id!r} is missing from the structure.")
    residues = [res for res in model[chain_id] if res.id[0] == " "]
    if not residues:
        raise ValueError(f"Designed chain {chain_id!r} has no protein residues.")
    for res in residues:
        if seq1(res.resname) not in "ACDEFGHIKLMNPQRSTVWY" or not all(
            atom in res for atom in ("N", "CA", "C", "O")
        ):
            raise ValueError(
                f"MPNN needs standard amino acids with N/CA/C/O in the designed "
                f"chain; check {chain_id}{res.id[1]}{res.id[2].strip()}."
            )
    return residues


def _design_positions(structure_path: Path, chain_id: str, fixed: set[int]):
    """Translate RFO's sequence-index mask to OFoldX component references."""
    from ofoldx.data.molecular import ComponentRef
    from ofoldx.io import read_structure
    from ofoldx.models.design import PositionSelection

    structure, _report = read_structure(structure_path)
    system = structure.system
    try:
        instance = system.instance_index(chain_id)
    except KeyError as exc:
        raise ValueError(f"Designed chain {chain_id!r} is missing from the structure.") from exc
    start = int(system.instance_component_ptr[instance].item())
    stop = int(system.instance_component_ptr[instance + 1].item())
    count = stop - start
    design = PositionSelection(
        components=tuple(
            ComponentRef(chain_id, index) for index in range(count) if index not in fixed
        )
    )
    return structure, design, count


def _candidate_sequence(candidate, chain_id: str) -> str:
    """Read one designed chain from an OFoldX coordinate-free candidate."""
    system = candidate.target.system
    try:
        instance = system.instance_index(chain_id)
    except KeyError as exc:
        raise ValueError(f"OFoldX design result is missing chain {chain_id!r}.") from exc
    start = int(system.instance_component_ptr[instance].item())
    stop = int(system.instance_component_ptr[instance + 1].item())
    component_names = system.text("component", system.component_symbol[start:stop])
    return "".join(seq1(name) for name in component_names.tolist())


def redesign(
    structure_file,
    fixed_residues_0idx,
    out_dir,
    num_seqs,
    temperature,
    seed,
    designed_chain="A",
):
    """Return ``(first candidate sequence, PDB path)`` for one RFO cycle.

    OFoldX's ProteinMPNN and LigandMPNN pipelines return coordinate-free
    sequence candidates. This adapter applies the first candidate's identities
    to the original PDB, preserving coordinates and all partner chains.
    """
    if num_seqs < 1 or temperature <= 0:
        raise ValueError("MPNN num_seqs and temperature must be positive.")
    source = Path(structure_file).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "mpnn_result.json").unlink(missing_ok=True)

    pdb_structure = PDBParser(QUIET=True).get_structure("input", str(source))
    residues = chain_residues(pdb_structure, designed_chain)
    fixed = set(fixed_residues_0idx)
    if any(index < 0 or index >= len(residues) for index in fixed):
        raise ValueError("MPNN fixed residue position is outside the designed chain.")

    model_type, model = settings()
    result_path = out / "mpnn_result.json"
    design_path = out / "mpnn_design.pdb"
    spec = {
        "structure_path": str(source),
        "designed_chain": designed_chain,
        "fixed_residues_0idx": sorted(fixed),
        "num_seqs": num_seqs,
        "temperature": temperature,
        "seed": seed,
        "model_type": model_type,
        "model": model,
    }

    if len(fixed) == len(residues):
        sequence = "".join(seq1(residue.resname) for residue in residues)
        shutil.copy2(source, design_path)
        spec["designed_residues"] = []
        candidates = []
    else:
        input_structure, design, component_count = _design_positions(
            source, designed_chain, fixed
        )
        if component_count != len(residues):
            raise ValueError(
                f"Chain {designed_chain!r} has {len(residues)} PDB residues but "
                f"{component_count} OFoldX components; refusing to shift the mask."
            )
        spec["designed_residues"] = [
            f"{designed_chain}{residue.id[1]}{residue.id[2].strip()}"
            for index, residue in enumerate(residues)
            if index not in fixed
        ]
        result = _load_pipeline(model_type, model).design(
            input_structure,
            design=design,
            num_candidates=num_seqs,
            temperature=temperature,
            seed=seed,
        )
        if not result.candidates:
            raise RuntimeError("OFoldX MPNN returned no sequence candidates.")
        candidates = []
        sequences = []
        for candidate in result.candidates:
            candidate_sequence = _candidate_sequence(candidate, designed_chain)
            if len(candidate_sequence) != len(residues):
                raise ValueError(
                    f"OFoldX returned {len(candidate_sequence)} residues for chain "
                    f"{designed_chain!r}; expected {len(residues)}."
                )
            for index in fixed:
                if candidate_sequence[index] != seq1(residues[index].resname):
                    raise ValueError(
                        f"OFoldX changed fixed residue {designed_chain}{residues[index].id[1]}"
                    )
            sequences.append(candidate_sequence)
            candidates.append({"sequence": candidate_sequence})

        # Preserve the original complex coordinates and apply only the first
        # candidate's sequence, matching RFO's existing cycle handoff behavior.
        sequence = sequences[0]
        for residue, amino_acid in zip(residues, sequence, strict=True):
            try:
                new_name = AA1_TO_3[amino_acid]
            except KeyError as exc:
                raise ValueError(
                    f"OFoldX returned unsupported amino-acid identity {amino_acid!r}."
                ) from exc
            if residue.resname != new_name:
                residue.resname = new_name
                for atom in list(residue):
                    if atom.name not in {"N", "CA", "C", "O", "OXT"}:
                        residue.detach_child(atom.id)
        writer = PDBIO()
        writer.set_structure(pdb_structure)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^Missing occupancy in atom .* written as blank$",
                category=PDBIOWarning,
                module=r"^Bio\.PDB\.PDBIO$",
            )
            writer.save(str(design_path))

    spec["candidates"] = candidates
    (out / "mpnn_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    result_path.write_text(
        json.dumps({"sequence": sequence, "structure_pdb": str(design_path)}, indent=2) + "\n"
    )
    return sequence, str(design_path)
