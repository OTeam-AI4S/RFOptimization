"""OFoldX Boltz-2 adapter used by RFO's structure-prediction branch."""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Superimposer

BOLTZ_MODEL_DEFAULT = "oteam/boltz2"


def settings():
    """Resolve the OFoldX Boltz-2 repository and check the package API."""
    model = os.environ.get("BOLTZ_MODEL", BOLTZ_MODEL_DEFAULT)
    try:
        import ofoldx
    except ImportError as exc:
        raise ImportError(
            "RFO's Boltz branch now uses OFoldX. Install OFoldX in the RFO "
            "Python environment and set BOLTZ_MODEL if needed; see docs/installation.md."
        ) from exc
    if not callable(getattr(ofoldx, "pipeline", None)):
        raise ImportError("The installed OFoldX package does not expose ofoldx.pipeline.")
    return model


@lru_cache(maxsize=4)
def _load_pipeline(model: str):
    import ofoldx
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ofoldx.pipeline("fold", model=model, device=device)
    if type(pipeline).__name__ != "Boltz2Pipeline":
        raise TypeError(
            f"BOLTZ_MODEL={model!r} must load an OFoldX Boltz2Pipeline; "
            f"loaded {type(pipeline).__name__}."
        )
    return pipeline


def _read_chain_msas(structure, msa_paths: Optional[Dict[str, str]]):
    """Build an OFoldX collection of independent, chain-targeted MSAs."""
    if not msa_paths:
        return None

    from ofoldx.data import TOKEN_ALPHABET, component_token_index
    from ofoldx.data.evidence import MSA, MsaCollection
    from ofoldx.io.msa import read_msa

    system = structure.system
    msas = []
    instance_ids = []
    for instance_id, path in msa_paths.items():
        try:
            instance = system.instance_index(instance_id)
        except KeyError as exc:
            raise ValueError(f"MSA supplied for missing chain {instance_id!r}.") from exc
        entity_index = int(system.instance_entity[instance].item())
        entity_id = system.entity_id(entity_index)
        msa_path = Path(path).expanduser().resolve()
        if not msa_path.is_file():
            raise FileNotFoundError(f"MSA does not exist: {msa_path}")
        msa = read_msa(msa_path, entity_id=entity_id)
        component_start = int(system.instance_component_ptr[instance].item())
        component_stop = int(system.instance_component_ptr[instance + 1].item())
        component_symbols = system.component_symbol[component_start:component_stop]
        query = "".join(
            TOKEN_ALPHABET[
                component_token_index(system.symbols[int(component_symbol)], "protein")
            ]
            for component_symbol in component_symbols.tolist()
        )
        if len(msa.unpaired_sequences[0]) != len(query):
            raise ValueError(
                f"MSA for chain {instance_id!r} has query width "
                f"{len(msa.unpaired_sequences[0])}, but the current sequence has "
                f"length {len(query)}"
            )

        # RFO's MPNN step may mutate this chain between cycles. Refresh the
        # query row to match the current structure while retaining homolog rows.
        # The query row itself has no insertion/deletion evidence after refresh.
        unpaired_deletions = msa.unpaired_deletion_counts.copy()
        paired_deletions = msa.paired_deletion_counts.copy()
        unpaired_deletions[0] = 0
        paired_deletions[0] = 0
        msa = MSA(
            entity_id=msa.entity_id,
            unpaired_sequences=(query, *msa.unpaired_sequences[1:]),
            unpaired_deletion_counts=unpaired_deletions,
            paired_sequences=(query, *msa.paired_sequences[1:]),
            paired_deletion_counts=paired_deletions,
        )
        msas.append(msa)
        instance_ids.append(instance_id)

    # First row is the query across all columns. Remaining rows retain each
    # chain's independent unpaired rows, without incorrectly pairing them.
    rows = [[0] * len(msas)]
    for column, msa in enumerate(msas):
        for row_index in range(1, len(msa.unpaired_sequences)):
            row = [-1] * len(msas)
            row[column] = row_index
            rows.append(row)
    # row_is_paired is a matrix; only the query row selects paired row zero.
    paired_matrix = np.zeros((len(rows), len(msas)), dtype=np.bool_)
    paired_matrix[0, :] = True
    return MsaCollection(
        tuple(msas),
        np.asarray(rows, dtype=np.int64),
        paired_matrix,
        target_instance_ids=tuple(instance_ids),
    )


def _prediction_metrics(prediction, chain_a="A", chain_b="B"):
    """Translate OFoldX typed confidence values to RFO's stable record keys."""
    scores = prediction.scores
    mean_plddt = scores.get("mean_plddt")
    if mean_plddt is None:
        plddt = prediction.metrics.get("plddt")
        mean_plddt_value = (
            float(np.nanmean(plddt.values)) if plddt is not None else float("nan")
        )
    else:
        mean_plddt_value = mean_plddt.value
    metrics = {
        "AF3_PTM": scores.get("ptm").value if "ptm" in scores else float("nan"),
        "AF3_iPTM": scores.get("iptm").value if "iptm" in scores else float("nan"),
        "AF3_iPAE": float("nan"),
        "AF3_design_ptm": float("nan"),
        "AF3_mean_pair_iptm": float("nan"),
        # Boltz reports complex pLDDT as mean_plddt (0--100); the per-token
        # values live in prediction.metrics["plddt"], not in scores["plddt"].
        "AF3_pLDDT": mean_plddt_value,
        "AF3_rmsd": float("nan"),
        "ranking_score": scores.get("confidence_score").value
        if "confidence_score" in scores
        else float("nan"),
        "AF3_Status": "Success",
    }

    pair_iptm = prediction.pair_metrics.get("chain_pair_iptm")
    if pair_iptm is not None:
        ids = list(pair_iptm.row_axis)
        if chain_a in ids and chain_b in ids:
            ab = float(pair_iptm.values[ids.index(chain_a), ids.index(chain_b)])
            ba = float(pair_iptm.values[ids.index(chain_b), ids.index(chain_a)])
            metrics["AF3_mean_pair_iptm"] = (ab + ba) / 2
        # The legacy Boltz confidence JSON exposed per-chain pTM values. In
        # OFoldX's directed chain-pair TM matrix, the diagonal is intra-chain
        # TM and carries the same chain-level signal.
        if chain_a in ids:
            metrics["AF3_design_ptm"] = float(
                pair_iptm.values[ids.index(chain_a), ids.index(chain_a)]
            )

    pae = prediction.pair_metrics.get("pae")
    if pae is not None:
        a_rows = [
            i for i, ref in enumerate(pae.row_axis) if _reference_chain(ref) == chain_a
        ]
        b_rows = [
            i for i, ref in enumerate(pae.row_axis) if _reference_chain(ref) == chain_b
        ]
        a_cols = [
            i
            for i, ref in enumerate(pae.column_axis)
            if _reference_chain(ref) == chain_a
        ]
        b_cols = [
            i
            for i, ref in enumerate(pae.column_axis)
            if _reference_chain(ref) == chain_b
        ]
        if a_rows and b_rows and a_cols and b_cols:
            block_ab = pae.values[np.ix_(a_rows, b_cols)]
            block_ba = pae.values[np.ix_(b_rows, a_cols)]
            metrics["AF3_iPAE"] = float((np.nanmin(block_ab) + np.nanmin(block_ba)) / 2)
    return metrics


def _reference_chain(reference):
    if isinstance(reference, tuple):
        return reference[0]
    # OFoldX axes use ComponentRef/AtomSite (``instance``), while older
    # adapters and serialized references may expose ``instance_id``.
    return getattr(reference, "instance", getattr(reference, "instance_id", None))


def _reference_component(reference):
    if isinstance(reference, tuple):
        return reference[1]
    # ComponentRef and AtomSite use ``component``; retain compatibility with
    # reference objects that name the field ``component_index``.
    return getattr(reference, "component", getattr(reference, "component_index", None))


def _read_rfo_boltz_structure(source: Path, scratch_dir: Path):
    """Read the coordinate-backed polymer sequence used by original RFO Boltz.

    The original Boltz adapter derived sequences from coordinate records, so
    unresolved residues listed only in PDB SEQRES were not sent to Boltz.
    OFoldX correctly preserves those declared residues by default. Remove only
    SEQRES from a temporary copy so its PDB reader derives each polymer from
    observed atoms, retaining the original coordinates, ligand records and
    connectivity in the input file.
    """
    from ofoldx.io import read_structure

    if source.suffix.lower() != ".pdb":
        return read_structure(source)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="ascii", suffix=".pdb", dir=scratch_dir, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with source.open(encoding="ascii", errors="replace") as input_file:
            for line in input_file:
                if not line.startswith("SEQRES"):
                    temporary.write(line)
    try:
        return read_structure(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fixed_residues(prediction, chain_id: str, threshold: float):
    """Match RFO's policy: fix high-pLDDT runs of at least five residues."""
    from ofoldx.data.molecular import EntityType

    confidence = prediction.metrics.get("plddt")
    if confidence is None:
        raise ValueError("OFoldX Boltz prediction did not provide a pLDDT metric.")

    # RFO's MPNN adapter numbers only standard amino-acid residues (the
    # residues Bio.PDB exposes with a blank hetero flag). OFoldX confidence
    # axes can also include modified polymer components and non-protein
    # components assigned to the same instance, so component count is not the
    # design-sequence length. Build the same ordered residue map explicitly.
    system = prediction.structure.system
    instance = int(system.instance_index([chain_id])[0])
    if instance < 0:
        raise ValueError(f"OFoldX Boltz prediction has no chain {chain_id!r}.")
    entity = int(system.instance_entity[instance])
    if int(system.entity_type[entity]) != int(EntityType.POLYMER):
        raise ValueError(f"OFoldX Boltz chain {chain_id!r} is not a polymer chain.")
    start = int(system.instance_component_ptr[instance])
    stop = int(system.instance_component_ptr[instance + 1])
    component_names = system.text("component", system.component_symbol[start:stop]).tolist()
    standard_amino_acids = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
    designed_components = [
        ordinal for ordinal, name in enumerate(component_names) if name in standard_amino_acids
    ]
    # Use the predicted polymer sequence as the index space. The source PDB
    # can contain only observed residues, while OFoldX retains unresolved
    # polymer components and Boltz predicts them too. The next MPNN step reads
    # this predicted structure, so its sequence indices must use these same
    # components rather than the source file's observed-residue count.
    chain_length = len(designed_components)
    if chain_length == 0:
        raise ValueError(f"OFoldX Boltz chain {chain_id!r} has no standard protein residues.")
    position_by_component = {component: index for index, component in enumerate(designed_components)}
    position_values = [[] for _ in range(chain_length)]
    for value, reference in zip(confidence.values, confidence.axis, strict=True):
        if _reference_chain(reference) != chain_id:
            continue
        component = _reference_component(reference)
        if component is None:
            raise ValueError(f"OFoldX pLDDT axis has no component index for chain {chain_id!r}.")
        position = position_by_component.get(int(component))
        if position is not None:
            position_values[position].append(float(value))
    if any(not values for values in position_values):
        found = sum(bool(values) for values in position_values)
        raise ValueError(
            f"OFoldX pLDDT covers {found} of {chain_length} standard protein residues "
            f"in chain {chain_id!r}."
        )
    per_position = [float(np.mean(values)) for values in position_values]
    good = [value > threshold for value in per_position]
    fixed: list[int] = []
    start = None
    for index, is_good in enumerate(good + [False]):
        if is_good and start is None:
            start = index
        elif not is_good and start is not None:
            if index - start >= 5:
                fixed.extend(range(start, index))
            start = None
    return fixed


def _calculate_ca_rmsd(first_file: str, second_file: str) -> Optional[float]:
    try:
        def parse(path):
            return MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)

        first = next(parse(first_file).get_structure("reference", first_file).get_models())
        second = next(parse(second_file).get_structure("prediction", second_file).get_models())

        def ca_atoms(model):
            return [
                residue["CA"]
                for chain in sorted(model, key=lambda c: c.id)
                for residue in chain
                if residue.has_id("CA")
            ]

        reference_atoms, predicted_atoms = ca_atoms(first), ca_atoms(second)
        if len(reference_atoms) != len(predicted_atoms):
            return None
        superimposer = Superimposer()
        superimposer.set_atoms(reference_atoms, predicted_atoms)
        return float(superimposer.rms)
    except Exception:
        return None


def run_boltz(
    structure_file: str,
    output_dir: str,
    num_samples: int,
    seed: int,
    msa_paths: Optional[Dict[str, str]] = None,
    cyclic_chains: Optional[List[str]] = None,
    template_plddt_threshold: float = 80,
) -> Tuple[Dict, Optional[str], List[int]]:
    """Run one OFoldX Boltz-2 fold and return RFO metrics, CIF path and mask."""
    start = time.time()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    source = Path(structure_file).expanduser().resolve()
    from ofoldx.io import write_structure
    from ofoldx.pipelines.generator import resolve_generator
    import torch

    structure, _report = _read_rfo_boltz_structure(source, target)
    msa = _read_chain_msas(structure, msa_paths)
    model_id = settings()
    pipeline = _load_pipeline(model_id)

    # Use OFoldX's staged Pipeline path so cyclic requests can be checked
    # against the bond-derived cyclic-period feature before sampling. The
    # current API separates prepared inputs from model features: prepare()
    # returns a PreparedInput, forward() consumes its features, and
    # postprocess() decodes the raw model output.
    generator = resolve_generator(seed, None, device=pipeline.processor.device)
    prepared = pipeline.prepare(
        structure,
        msa=msa,
        generator=None if seed is None else np.random.default_rng(seed),
    )
    features = prepared.features
    for chain_id in cyclic_chains or ():
        instance = structure.system.instance_index(chain_id)
        chain_index = instance
        cyclic = features.cyclic_period
        if cyclic is None or not bool(
            (cyclic[0][features.chain_index[0] == chain_index] > 0).any()
        ):
            raise ValueError(
                f"Cyclic chain {chain_id!r} has no terminal covalent closure in the "
                "input structure; OFoldX needs that bond to enable cyclic conditioning."
            )
    with torch.inference_mode():
        model_output = pipeline.forward(
            features,
            num_samples=num_samples,
            generator=generator,
        )
    result = pipeline.postprocess(prepared, model_output)
    if not result.predictions:
        raise RuntimeError("OFoldX Boltz-2 returned no structure predictions.")

    prediction = result.predictions[0]
    tag = f"{source.stem}_{target.name}"
    cif_path = target / f"{tag}.cif"
    # Model outputs may not carry archive author numbering, even though the
    # prepared input has stable instance symbols. Assign only the identifiers
    # required by mmCIF so the result can be handed back to RFO and MPNN.
    write_structure(prediction.structure, cif_path, identity="assign")
    fixed = _fixed_residues(prediction, "A", template_plddt_threshold)
    metrics = _prediction_metrics(prediction)
    # The CIF writer can omit B-factor values that Bio.PDB requires while
    # parsing mmCIF. RFO converts this file to PDB immediately afterward;
    # calculate RMSD there, where coordinate-only parsing is reliable.
    metrics["AF3_rmsd"] = float("nan")
    metrics["method"] = "Boltz"
    metrics["Inference_Cost_Sec"] = time.time() - start
    return metrics, str(cif_path), fixed
