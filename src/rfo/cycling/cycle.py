#!/usr/bin/env python
"""Cycling driver. Each cycle:

    1. With probability ``backprop_fraction`` run the configured gradient optimizer
       (``backprop/optimize.py``); otherwise run Boltz on the current sequence.
       Both produce a structure for the current design.
    2. Redesign the binder chain on that structure with MPNN.
    3. The redesigned structure becomes the input to the next cycle.

The Protenix gradient path, OFoldX Boltz-2, and MPNN run in the RFO driver's
Python environment. See ``docs/installation.md`` for setup.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from Bio.PDB import MMCIFParser, PDBParser

from rfo.cycling.boltz import boltz_runner

# Standard 3-letter -> 1-letter map (hardcoded to avoid biopython-version drift;
# older Bio.PDB.Polypeptide lacks ``protein_letters_3to1``).
protein_letters_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


# ============================================================
# Configuration
# ============================================================
RFO_SRC = Path(__file__).resolve().parents[2]
BACKPROP_DIR = RFO_SRC / "rfo" / "backprop"


# ============================================================
# Small structure helpers (pure Bio/biotite — no torch needed)
# ============================================================
def _structure_parser(path: str):
    return MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)


def get_chain_sequence(path: str, chain_id: str = "A") -> str:
    """Return the one-letter sequence of ``chain_id`` from a PDB/CIF file."""
    structure = _structure_parser(path).get_structure("s", path)
    model = next(structure.get_models())
    seq = []
    if chain_id in [c.id for c in model]:
        chain = model[chain_id]
    else:
        raise ValueError(f"Required chain {chain_id!r} is missing from {path}")
    for residue in chain:
        if residue.id[0] != " ":
            continue  # skip hetero/water
        resname = residue.get_resname().strip().upper()
        seq.append(protein_letters_3to1.get(resname, "X"))
    return "".join(seq)


def structure_to_af3_sequences(path: str) -> List[Dict]:
    """Build protein, nucleic-acid or CCD ligand blocks from distinct chains."""
    structure = _structure_parser(path).get_structure("s", path)
    blocks = []
    alphabets = {
        "protein": protein_letters_3to1,
        "dna": {"DA": "A", "DC": "C", "DG": "G", "DT": "T"},
        "rna": {"A": "A", "C": "C", "G": "G", "U": "U"},
    }
    for chain in next(structure.get_models()):
        residues = [r for r in chain if r.resname.strip() not in {"HOH", "WAT"}]
        if not residues:
            continue
        names = [r.resname.strip().upper() for r in residues]
        for kind, alphabet in alphabets.items():
            if all(name in alphabet for name in names):
                blocks.append(
                    {
                        kind: {
                            "id": chain.id,
                            "sequence": "".join(alphabet[name] for name in names),
                        }
                    }
                )
                break
        else:
            if len(residues) != 1:
                raise ValueError(
                    f"Cannot infer a single entity for chain {chain.id}; give each "
                    "ligand entity its own chain ID in the input structure."
                )
            blocks.append({"ligand": {"id": chain.id, "ccd": names[0]}})
    return blocks


def force_copy(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(dst):
        try:
            os.chmod(dst, 0o777)
            os.remove(dst)
        except OSError:
            pass
    shutil.copy2(src, dst)


def _cif_to_pdb_ofoldx(cif_path: str, pdb_path: str) -> None:
    """Use OFoldX's canonical reader and writer for modern OFoldX CIF files."""
    from ofoldx.io import read_structure, write_structure

    structure, _report = read_structure(cif_path)
    write_structure(structure, pdb_path, format="pdb", identity="assign")


def _cif_to_pdb_biotite(cif_path: str, pdb_path: str) -> None:
    """Convert general mmCIF with Biotite while preserving label chain/residue IDs."""
    import numpy as np
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif = pdbx.CIFFile.read(cif_path)
    atom_array = pdbx.get_structure(cif, model=1)
    cif_dict = MMCIF2Dict(cif_path)

    def values(key):
        value = cif_dict.get(key, [])
        return value if isinstance(value, list) else [value]

    label_chains = values("_atom_site.label_asym_id")
    label_residues = values("_atom_site.label_seq_id")
    if len(label_chains) == len(atom_array) and len(label_residues) == len(atom_array):
        atom_array.chain_id = np.asarray(label_chains, dtype=str)
        atom_array.res_id = np.asarray(
            [
                int(value) if value not in {".", "?"} else index + 1
                for index, value in enumerate(label_residues)
            ],
            dtype=int,
        )
    fixed = atom_array.res_name.copy()
    for i, name in enumerate(fixed):
        if len(name) > 3:
            fixed[i] = name[:3]
    atom_array.res_name = fixed
    out = pdb.PDBFile()
    pdb.set_structure(out, atom_array)
    out.write(pdb_path)


def _cif_to_pdb_biopython(cif_path: str, pdb_path: str) -> None:
    from Bio.PDB import MMCIFParser, PDBIO

    structure = MMCIFParser(QUIET=True).get_structure("s", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)


def _fill_missing_pdb_occupancies(pdb_path: str) -> None:
    """Write the PDB default occupancy for atoms whose source CIF omitted it.

    OFoldX can write valid CIF atoms without occupancy values. Its PDB writer
    preserves those as blank occupancy columns, but Bio.PDB cannot parse a
    disordered atom with a missing occupancy (it compares the occupancy while
    selecting an alternate location). A predicted single-model atom with no
    reported occupancy is conventionally represented with occupancy 1.00.
    """
    path = Path(pdb_path)
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    normalized = []
    for raw_line in lines:
        ending = "\n" if raw_line.endswith("\n") else ""
        line = raw_line[:-1] if ending else raw_line
        if line.startswith(("ATOM  ", "HETATM")):
            line = line.ljust(80)
            if not line[54:60].strip():
                line = f"{line[:54]}  1.00{line[60:]}"
        normalized.append(line + ending)
    path.write_text("".join(normalized), encoding="ascii")


def cif_to_pdb(
    cif_path: str, pdb_path: str, *, assume_full_occupancy: bool = False
) -> bool:
    """Convert CIF to PDB through format-aware writers, retaining generic fallbacks."""
    converters = (
        ("OFoldX", _cif_to_pdb_ofoldx),
        ("Biotite", _cif_to_pdb_biotite),
        ("Bio.PDB", _cif_to_pdb_biopython),
    )
    for name, converter in converters:
        try:
            converter(cif_path, pdb_path)
            if assume_full_occupancy:
                _fill_missing_pdb_occupancies(pdb_path)
            return True
        except Exception as error:  # noqa: BLE001
            print(f"{name} CIF-to-PDB conversion failed: {error}")
    return False


def get_residues_within_cutoff(
    path: str, chain_prot: str = "A", chain_lig: str = "B", cutoff: float = 8.0
) -> List[int]:
    """1-indexed residues of ``chain_prot`` with any atom within ``cutoff`` Å
    of ``chain_lig`` (used to fix interface residues during redesign)."""
    structure = _structure_parser(path).get_structure("s", path)
    model = next(structure.get_models())
    chains = {c.id: c for c in model}
    if chain_prot not in chains or chain_lig not in chains:
        return []
    lig_atoms = [a for r in chains[chain_lig] for a in r]
    contacts = []
    for i, residue in enumerate(
        (r for r in chains[chain_prot] if r.id[0] == " "), start=1
    ):
        hit = any((a - la) <= cutoff for a in residue for la in lig_atoms)
        if hit:
            contacts.append(i)
    return sorted(contacts)


def template_process(
    cif_path: str, plddt_threshold: float = 80, min_continuous_length: int = 5
) -> List[int]:
    """Return 0-indexed chain-A residues whose mean pLDDT (CIF B-factor) is
    above ``plddt_threshold`` and that fall in a contiguous run of at least
    ``min_continuous_length`` residues."""
    structure = MMCIFParser(QUIET=True).get_structure("confidence", cif_path)
    model = next(structure.get_models())
    if "A" not in model:
        raise ValueError("Boltz output is missing designed chain A.")
    residues = [r for r in model["A"] if r.id[0] == " "]
    # Masks use positions in chain A, not PDB author residue numbers.
    good = [
        i
        for i, res in enumerate(residues)
        if sum(atom.bfactor for atom in res) / len(res) > plddt_threshold
    ]

    # keep contiguous runs only
    out: List[int] = []
    run: List[int] = []
    for rid in good:
        if run and rid == run[-1] + 1:
            run.append(rid)
        else:
            if len(run) >= min_continuous_length:
                out.extend(run)
            run = [rid]
    if len(run) >= min_continuous_length:
        out.extend(run)
    return out


def process_history_metrics(
    history_path: str, status_key: str = "PROTENIX_Status"
) -> Tuple[Dict, Optional[int]]:
    """Parse the optimizer ``history.json`` and return (metrics, best_step).

    ``best_step`` is the ``step`` field of the lowest-loss entry (used to find
    ``modelhub_pred/<step>.cif``)."""
    metrics: Dict = {status_key: "Failed"}
    try:
        with open(history_path) as f:
            history = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"cannot read history {history_path}: {e}")
        return metrics, None
    if not history:
        return metrics, None

    def loss_of(entry):
        for k in ("loss_prop", "loss"):
            if k in entry and entry[k] is not None:
                return entry[k]
        return float("inf")

    best = min(history, key=loss_of)
    metrics = {
        status_key: "Success",
        "step": best.get("step"),
        "loss": loss_of(best),
        "ipae_min": best.get("ipae_min", float("nan")),
        "ipae_mean": best.get("ipae_mean", float("nan")),
        "iptm": best.get("iptm", float("nan")),
        "plddt_mean": best.get("plddt_mean", float("nan")),
        "substrate_plddt": best.get("substrate_plddt", float("nan")),
    }
    return metrics, best.get("step")


# ============================================================
# Branch: Boltz structure prediction
# ============================================================
def run_boltz_branch(
    structure_file: str,
    cycle: int,
    output_dir: str,
    num_samples: int,
    template_plddt_threshold: int,
    cyclic_chains: Optional[List[str]],
    msa_paths: Optional[Dict] = None,
    seed: int = 0,
) -> Tuple[Dict, Optional[str], List[int]]:
    """Predict the current design with OFoldX Boltz-2. Returns (metrics, pdb_path,
    fixed_residues_0idx)."""
    target_dir = os.path.join(output_dir, f"recycle_{cycle + 1}")
    os.makedirs(target_dir, exist_ok=True)
    metrics, cif_out, fixed = boltz_runner.run_boltz(
        structure_file,
        target_dir,
        num_samples=num_samples,
        seed=seed,
        msa_paths=msa_paths,
        cyclic_chains=cyclic_chains,
        template_plddt_threshold=template_plddt_threshold,
    )
    if cif_out is None:
        return metrics, None, []
    pdb_out = str(Path(cif_out).with_suffix(".pdb"))
    if not cif_to_pdb(cif_out, pdb_out, assume_full_occupancy=True):
        return metrics, None, []
    rmsd = boltz_runner._calculate_ca_rmsd(pdb_out, structure_file)
    metrics["boltz_rmsd"] = float("nan") if rmsd is None else rmsd
    return metrics, pdb_out, fixed


# ============================================================
# Branch: configured gradient/MCMC sequence optimization
# ============================================================
def run_gradient_branch(
    base_config: str,
    structure_file: str,
    cycle: int,
    output_dir: str,
    cyclic: bool,
    loss: Optional[str] = None,
    msa_paths: Optional[Dict] = None,
    paired_msa_paths: Optional[Dict] = None,
    templates: Optional[Dict] = None,
) -> Tuple[Dict, Optional[str], List[int]]:
    """Optimize the current design with the configured gradient folding model. Returns
    (metrics, pdb_path, fixed_residues_0idx)."""
    target_dir = os.path.join(output_dir, f"recycle_{cycle + 1}")
    os.makedirs(target_dir, exist_ok=True)
    stem = Path(structure_file).stem

    # Generate a per-cycle config from the user template: override input +
    # output_path (and any optional loss / msa / template) so the optimizer
    # reads our structure and writes here.
    with open(base_config) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    cfg["input"] = os.path.abspath(structure_file)
    cfg["output_path"] = os.path.abspath(target_dir)
    if loss:
        cfg["loss_type"] = loss
    if msa_paths:
        cfg["msa_paths"] = msa_paths
    if paired_msa_paths:
        cfg["paired_msa_paths"] = paired_msa_paths
    if templates:
        cfg["fix_template_dict"] = templates

    # Per-cycle files belong to the run, not a potentially read-only install.
    config_file = Path(target_dir).resolve() / "protenix_config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    cmd = [
        sys.executable,
        str(BACKPROP_DIR / "optimize.py"),
        "--config_name",
        config_file.stem,
        "--config_path",
        str(config_file.parent.resolve()),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(RFO_SRC), env.get("PYTHONPATH", "")) if value
    )
    print("\nPROTENIX optimizer cmd:\n" + " ".join(cmd) + "\n")
    start = time.time()
    try:
        subprocess.run(cmd, env=env, check=True)
        success, cost = True, time.time() - start
    except subprocess.CalledProcessError as e:
        print(f"PROTENIX optimizer failed (rc={e.returncode})")
        success, cost = False, time.time() - start

    metrics = {"method": "PROTENIX", "Inference_Cost_Sec": cost}
    if not success:
        metrics["PROTENIX_Status"] = "Failed"
        return metrics, None, []

    # The optimizer writes to <output_path>/<stem><suffix>/ — find it.
    hist_candidates = sorted(
        glob.glob(os.path.join(target_dir, "*", "history.json")),
        key=os.path.getmtime,
    )
    if not hist_candidates:
        print(f"PROTENIX: no history.json under {target_dir}")
        metrics["PROTENIX_Status"] = "Failed"
        return metrics, None, []
    opt_dir = os.path.dirname(hist_candidates[-1])

    gradient_metrics, best_step = process_history_metrics(
        hist_candidates[-1], status_key="PROTENIX_Status"
    )
    metrics.update(gradient_metrics)

    # Locate the predicted structure for the best step.
    pred_dir = os.path.join(opt_dir, "modelhub_pred")
    cif_src = os.path.join(pred_dir, f"{best_step}.cif")
    if not os.path.exists(cif_src):
        cifs = sorted(glob.glob(os.path.join(pred_dir, "*.cif")), key=os.path.getmtime)
        if not cifs:
            print(f"PROTENIX: no predicted CIF in {pred_dir}")
            metrics["PROTENIX_Status"] = "Failed"
            return metrics, None, []
        cif_src = cifs[-1]

    cif_out = os.path.join(target_dir, f"{stem}.cif")
    shutil.copy2(cif_src, cif_out)
    pdb_out = cif_out.replace(".cif", ".pdb")
    if not cif_to_pdb(cif_out, pdb_out, assume_full_occupancy=True):
        return metrics, None, []

    # Fix protein residues in contact with the partner/ligand chain B.
    fixed = [i - 1 for i in get_residues_within_cutoff(pdb_out, "A", "B", 8.0)]
    return metrics, pdb_out, fixed


# ============================================================
# MPNN redesign through OFoldX's ProteinMPNN/LigandMPNN Pipelines
# ============================================================
def run_mpnn(
    structure_file,
    fixed_residues_0idx,
    out_dir,
    num_seqs,
    temperature,
    seed,
    designed_chain="A",
):
    from rfo.cycling.mpnn.mpnn_worker import redesign

    return redesign(
        structure_file,
        fixed_residues_0idx,
        out_dir,
        num_seqs,
        temperature,
        seed,
        designed_chain,
    )


# ============================================================
# Master cycling loop
# ============================================================
def validate_dependencies(args):
    """Check OFoldX and the model installations needed by this run."""
    from rfo.cycling.boltz import boltz_runner
    from rfo.cycling.mpnn.mpnn_worker import settings

    settings()

    if args.backprop_fraction > 0:
        with open(args.optimizer_config) as f:
            config = yaml.safe_load(f)
        if not config.get("protenix_model"):
            raise ValueError("Protenix config must specify protenix_model")
        loss = args.loss or config.get("loss_type", "")
        if loss not in {
            "pae_interface_mean",
            "pae_interface_min",
            "pae_mean",
            "pde_mean",
            "plddt_mean",
            "iptm",
        }:
            raise ValueError(
                "Protenix cycling requires a supported confidence objective to produce a structure for MPNN."
            )
    if args.backprop_fraction < 1:
        boltz_runner.settings()


def run_loop(args) -> None:
    validate_dependencies(args)
    random.seed(args.seed)
    input_dir = os.path.join(args.out_dir, "inputs")
    output_dir = args.out_dir
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(args.input_structure).stem
    current = os.path.join(input_dir, f"{stem}_0.pdb")
    if args.input_structure.endswith(".cif"):
        if not cif_to_pdb(args.input_structure, current):
            raise RuntimeError("Could not convert the input complex to PDB.")
    else:
        force_copy(args.input_structure, current)

    cyclic_chains = ["A"] if args.cyclic else None
    records: List[Dict] = []
    records_path = Path(output_dir) / f"{stem}_records.json"
    records_path.write_text("[]\n")
    gradient_backend = "Protenix"
    cycle, attempts = 0, 0
    max_attempts = args.total_cycles * 3

    while cycle < args.total_cycles and attempts < max_attempts:
        attempts += 1
        use_protenix = random.random() < args.backprop_fraction
        print(
            f"\n=== cycle {cycle}/{args.total_cycles} "
            f"(attempt {attempts}/{max_attempts}) "
            f"method={gradient_backend if use_protenix else 'Boltz'} ==="
        )
        t0 = time.time()

        if use_protenix:
            metrics, struct_path, fixed = run_gradient_branch(
                args.optimizer_config,
                current,
                cycle,
                output_dir,
                args.cyclic,
                loss=args.loss,
                msa_paths=args.msa_paths,
                paired_msa_paths=args.paired_msa_paths,
                templates=args.templates,
            )
        else:
            metrics, struct_path, fixed = run_boltz_branch(
                current,
                cycle,
                output_dir,
                args.num_samples,
                args.template_plddt_threshold,
                cyclic_chains,
                msa_paths=args.msa_paths,
                seed=args.seed + cycle,
            )

        if struct_path is None or not os.path.exists(struct_path):
            print(f"structure step failed at cycle {cycle}; retrying")
            continue

        # MPNN redesign of the binder chain.
        print("--- MPNN redesign ---")
        mpnn_dir = os.path.join(output_dir, f"recycle_{cycle + 1}", "mpnn")
        seq, redesigned = run_mpnn(
            struct_path,
            fixed,
            mpnn_dir,
            args.mpnn_num_seqs,
            args.mpnn_temperature,
            args.seed + cycle,
        )
        if redesigned is None or not os.path.exists(redesigned):
            print(f"MPNN failed at cycle {cycle}; retrying")
            continue

        # Hand-off to next cycle.
        next_input = os.path.join(input_dir, f"{stem}_{cycle + 1}.pdb")
        if redesigned.endswith(".cif"):
            if not cif_to_pdb(redesigned, next_input, assume_full_occupancy=True):
                raise RuntimeError("Could not convert the MPNN design to PDB.")
        else:
            force_copy(redesigned, next_input)
        current = next_input

        metrics.update(
            {
                "cycle": cycle,
                "mpnn_sequence": seq,
                "wall_time_sec": round(time.time() - t0, 2),
            }
        )
        records.append(metrics)
        records_path.write_text(json.dumps(records, indent=2) + "\n")
        cycle += 1

    if cycle < args.total_cycles:
        print(
            f"only {cycle}/{args.total_cycles} cycles completed "
            f"after {attempts} attempts"
        )

    with open(os.path.join(output_dir, f"{stem}_records.json"), "w") as f:
        json.dump(records, f, indent=2)
    if cycle < args.total_cycles:
        raise RuntimeError(
            f"Only {cycle}/{args.total_cycles} cycles completed; see saved records."
        )
    print(f"\nDone. Records → {os.path.join(output_dir, f'{stem}_records.json')}")
