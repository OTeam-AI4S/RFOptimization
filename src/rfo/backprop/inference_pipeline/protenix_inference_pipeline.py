"""RFO adapter for OFoldX's Protenix sequence-optimization path."""

from __future__ import annotations

from pathlib import Path

import torch


class ProtenixInferencePipeline:
    """Keep RFO's objective API while delegating folding to OFoldX Protenix."""

    backend = "protenix"

    def __init__(
        self,
        model_name: str,
        *,
        device: str | torch.device,
        n_recycles: int = 10,
        num_sampling_steps: int | None = None,
    ) -> None:
        try:
            from ofoldx.auto import AutoModel, AutoProcessor
            from ofoldx.models.protenix.model import ProtenixModel
            from ofoldx.models.protenix.model import ProtenixTemplateModel
            from ofoldx.models.protenix.pipeline import (
                ProtenixPipeline,
                ProtenixTemplatePipeline,
            )
            from ofoldx.models.protenix.processor import (
                ProtenixProcessor,
                ProtenixTemplateProcessor,
            )
            from ofoldx.templates import (
                Template,
                TemplateAlignment,
                chain_coordinates,
            )
        except ImportError as error:
            raise ImportError(
                "The RFO Protenix adapter requires a compatible OFoldX installation "
                "with its current model and template APIs."
            ) from error

        self.device = torch.device(device)
        self.n_recycles = n_recycles
        self.num_sampling_steps = num_sampling_steps
        self.model_name = model_name
        self._template_type = (Template, TemplateAlignment)
        self._chain_coordinates = chain_coordinates
        self._template_sources = []
        self.model = AutoModel.from_pretrained(model_name, device=self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, device=self.device)
        if not isinstance(self.model, ProtenixModel) or not isinstance(
            self.processor, ProtenixProcessor
        ):
            raise TypeError(
                f"{model_name!r} must resolve to an OFoldX Protenix model and processor"
            )
        self.template_capable = isinstance(self.model, ProtenixTemplateModel) and isinstance(
            self.processor, ProtenixTemplateProcessor
        )
        if isinstance(self.model, ProtenixTemplateModel) != isinstance(
            self.processor, ProtenixTemplateProcessor
        ):
            raise TypeError(
                "Template-capable Protenix models and processors must be loaded as a pair"
            )
        self.model.eval()
        # RFO differentiates with respect to its temporary sequence tensor only.
        # Freezing the model weights avoids allocating and accumulating parameter
        # gradients while preserving gradients through the model to that input.
        self.model.requires_grad_(False)
        if self.template_capable:
            self.pipeline = ProtenixTemplatePipeline(
                model=self.model, processor=self.processor
            )
        else:
            self.pipeline = ProtenixPipeline(model=self.model, processor=self.processor)
        from atomworks.ml.encoding_definitions import AF3SequenceEncoding
        from atomworks.constants import DICT_THREE_TO_ONE

        self.encoding = AF3SequenceEncoding()
        self._aa_order = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_three = {one: three for three, one in DICT_THREE_TO_ONE.items()}
        self._encoded_aa = torch.tensor(
            [int(self.encoding.encode(aa_to_three[amino_acid])) for amino_acid in self._aa_order],
            dtype=torch.long,
        )
        self.has_confidence = True

    def prepare_fold(self, structure_path: str | Path):
        """Parse a structure and prepare OFoldX input tensors."""
        return self._prepare_fold(self._read_structure(Path(structure_path)))

    @staticmethod
    def _read_structure(path: Path):
        """Return the Structure from OFoldX's ``(Structure, ReadReport)`` result."""
        from ofoldx.io import read_structure

        structure, _report = read_structure(path)
        return structure

    def configure_templates(
        self,
        structure_path: str | Path,
        *,
        fix_template_dict=None,
        self_template_selection_syntax: str | None = None,
    ) -> None:
        """Cache RFO template evidence for the initial structure and all candidates."""

        self._template_sources = []
        if not fix_template_dict and not self_template_selection_syntax:
            return
        if not self.template_capable:
            raise ValueError(
                "RFO templates require an OFoldX Protenix template-capable model and processor"
            )

        if fix_template_dict:
            for target_chain, template_path in fix_template_dict.items():
                source = self._read_structure(Path(template_path))
                for instance_index in range(len(source.system.instance_symbol)):
                    if not self._is_protein_instance(source.system, instance_index):
                        continue
                    source_chain = source.system.text(
                        "instance", source.system.instance_symbol[instance_index : instance_index + 1]
                    )[0]
                    self._template_sources.append(
                        (source, str(target_chain), str(source_chain), None)
                    )

        if self_template_selection_syntax:
            from atomworks.io.parser import parse
            from rfo.backprop.tools.af3 import apply_template_selection

            parsed = parse(Path(structure_path), hydrogen_policy="remove")
            atom_array = (
                parsed["assemblies"]["1"][0]
                if "assemblies" in parsed
                else parsed["asym_unit"][0]
            )
            selected = apply_template_selection(atom_array, self_template_selection_syntax)
            selected_mask = selected.get_annotation("is_input_file_templated")
            selected_residues = {}
            for chain_id, residue_id, is_selected in zip(
                selected.chain_id, selected.res_id, selected_mask
            ):
                if is_selected:
                    selected_residues.setdefault(str(chain_id), set()).add(int(residue_id))
            source = self._read_structure(Path(structure_path))
            for chain_id, residue_ids in selected_residues.items():
                self._template_sources.append((source, chain_id, chain_id, residue_ids))

    def _templates_for(self, structure):
        if not self._template_sources:
            return ()
        Template, TemplateAlignment = self._template_type
        system = structure.system
        instance_names = system.text("instance", system.instance_symbol)
        target_instances = {
            str(instance_name): index for index, instance_name in enumerate(instance_names)
        }
        templates = []
        for source, target_chain, source_chain, selected_residues in self._template_sources:
            if target_chain not in target_instances:
                # A configured target may be absent from a candidate after other
                # chain filtering; it simply contributes no evidence to that fold.
                continue
            target_index = target_instances[target_chain]
            target_entity_index = int(system.instance_entity[target_index].item())
            if not self._is_protein_instance(system, target_index):
                continue
            source_index = int(source.system.instance_index([source_chain])[0])
            if source_index < 0 or not self._is_protein_instance(source.system, source_index):
                continue
            target_start, target_stop = system.instance_component_ptr[
                target_index : target_index + 2
            ].tolist()
            source_start, source_stop = source.system.instance_component_ptr[
                source_index : source_index + 2
            ].tolist()
            target_ids = structure.component_author_number[target_start:target_stop].tolist()
            source_ids = source.component_author_number[source_start:source_stop].tolist()
            source_positions = {}
            for index, residue_id in enumerate(source_ids):
                if selected_residues is None or int(residue_id) in selected_residues:
                    source_positions.setdefault(int(residue_id), index)

            pairs = []
            previous_source_index = -1
            for target_position, residue_id in enumerate(target_ids):
                source_position = source_positions.get(int(residue_id))
                if source_position is not None and source_position > previous_source_index:
                    pairs.append((target_position, source_position))
                    previous_source_index = source_position
            if not pairs:
                continue
            alignment = TemplateAlignment(
                target_entity_id=str(
                    system.text("entity", system.entity_symbol[target_entity_index : target_entity_index + 1])[0]
                ),
                target_instance_id=target_chain,
                template_instance_id=source_chain,
                residue_pairs=tuple(pairs),
            )
            templates.append(
                Template(
                    coordinates=self._chain_coordinates(source, (source_chain,)),
                    alignments=(alignment,),
                )
            )
        return tuple(templates)

    @staticmethod
    def _is_protein_instance(system, instance_index: int) -> bool:
        import numpy as np

        from ofoldx.data.molecular import MonomerType
        from ofoldx.data.tokenization import CLASS_MONOMER, token_classes

        start, stop = system.instance_component_ptr[instance_index : instance_index + 2]
        monomer = CLASS_MONOMER[token_classes(system).token[int(start) : int(stop)]]
        return bool(monomer.size and np.all(monomer == MonomerType.PROTEIN))

    def _prepare_fold(self, structure, *, msa=None):
        templates = self._templates_for(structure) if hasattr(structure, "system") else ()
        if templates and not self.template_capable:
            raise ValueError("The selected Protenix checkpoint cannot consume template evidence")
        if self.template_capable:
            prepared = self.pipeline.prepare(structure, msa=msa, templates=templates)
        else:
            prepared = self.pipeline.prepare(structure, msa=msa)
        # Preserve the canonical context as well as the model features. RFO
        # needs the original structure metadata to map token chains back to
        # the chain labels used by its interface objectives.
        return prepared.prepared

    def prepare_fold_with_msa(
        self, structure_path: str | Path, msa_paths, paired_msa_paths=None
    ):
        """Prepare an input with RFO's per-chain A3M/FASTA paths."""

        if not msa_paths and paired_msa_paths:
            raise ValueError("paired_msa_paths requires matching per-chain msa_paths")
        if not msa_paths:
            return self.prepare_fold(structure_path)

        import numpy as np
        from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
        from ofoldx.msa import MSA, MsaCollection, read_msa

        path = Path(structure_path)
        structure = self._read_structure(path)
        system = structure.system if hasattr(structure, "system") else structure
        prepared_input = self._prepare_fold(structure)
        base_input = prepared_input.features
        valid_tokens = base_input.token_mask[0].bool()
        chain_indices = base_input.chain_index[0, valid_tokens]
        protein_tokens = base_input.token_has_esm[0, valid_tokens].bool()
        vocab = base_input.token_features[0, valid_tokens, 384 : 384 + len(TOKENS)]
        index_to_symbol = {index: symbol for symbol, index in TOKEN_INDEX.items()}

        chain_context = {}
        for chain_id, msa_path in msa_paths.items():
            instance_index = int(system.instance_index([str(chain_id)])[0])
            if instance_index < 0:
                raise ValueError(f"RFO MSA mapping references missing Protenix chain {chain_id!r}") from None
            entity_index = int(system.instance_entity[instance_index].item())
            entity_id = str(
                system.text("entity", system.entity_symbol[entity_index : entity_index + 1])[0]
            )
            chain_mask = (chain_indices == instance_index) & protein_tokens
            query = "".join(
                index_to_symbol[int(index)]
                for index in vocab[chain_mask].argmax(dim=-1).tolist()
            )
            msa = read_msa(msa_path, entity_id=entity_id)
            msa = self._resize_msa_columns(msa, len(query), chain_id)
            if len(msa.unpaired_sequences[0]) != len(query):
                raise ValueError(
                    f"MSA for chain {chain_id!r} has query width {len(msa.unpaired_sequences[0])}, "
                    f"but the current sequence has length {len(query)}"
                )
            # RFO mutates the query while retaining the homolog alignments. Keep
            # the alignment columns and update row zero to the current sequence.
            msa = MSA(
                entity_id=msa.entity_id,
                unpaired_sequences=(query, *msa.unpaired_sequences[1:]),
                unpaired_deletion_counts=msa.unpaired_deletion_counts,
                paired_sequences=(query, *msa.paired_sequences[1:]),
                paired_deletion_counts=msa.paired_deletion_counts,
            )
            chain_context[str(chain_id)] = (entity_id, query, msa)

        if paired_msa_paths:
            if set(paired_msa_paths) != set(msa_paths):
                raise ValueError(
                    "paired_msa_paths must provide a paired alignment for every chain in msa_paths"
                )
            columns = []
            target_instances = []
            for chain_id in msa_paths:
                entity_id, query, independent_msa = chain_context[str(chain_id)]
                source_msa = read_msa(paired_msa_paths[chain_id], entity_id=entity_id)
                source_msa = self._resize_msa_columns(source_msa, len(query), chain_id)
                columns.append(
                    MSA(
                        entity_id=entity_id,
                        unpaired_sequences=independent_msa.unpaired_sequences,
                        unpaired_deletion_counts=independent_msa.unpaired_deletion_counts,
                        paired_sequences=(query, *source_msa.unpaired_sequences[1:]),
                        paired_deletion_counts=source_msa.unpaired_deletion_counts.copy(),
                    )
                )
                target_instances.append(str(chain_id))
            max_paired_rows = max(len(msa.paired_sequences) for msa in columns)
            row_pairing = np.full((max_paired_rows, len(columns)), -1, dtype=np.int64)
            row_is_paired = np.ones_like(row_pairing, dtype=np.bool_)
            row_pairing[0] = 0
            for row_index in range(1, max_paired_rows):
                for column_index, msa in enumerate(columns):
                    if row_index < len(msa.paired_sequences):
                        row_pairing[row_index, column_index] = row_index
            collection = MsaCollection(
                msas=tuple(columns),
                row_pairing=row_pairing,
                row_is_paired=row_is_paired,
                target_instance_ids=tuple(target_instances),
            )
            return self._prepare_fold(structure, msa=collection)

        columns = [context[2] for context in chain_context.values()]
        target_instances = list(chain_context)

        # RFO supplies independent per-chain alignments, so homolog rows remain
        # unpaired across chains. The query is the only shared paired row.
        pairing_rows = [np.zeros((len(columns),), dtype=np.int64)]
        paired_rows = [np.ones((len(columns),), dtype=np.bool_)]
        for column_index, msa in enumerate(columns):
            for row_index in range(1, len(msa.unpaired_sequences)):
                pairing = np.full((len(columns),), -1, dtype=np.int64)
                paired = np.zeros((len(columns),), dtype=np.bool_)
                pairing[column_index] = row_index
                pairing_rows.append(pairing)
                paired_rows.append(paired)
        collection = MsaCollection(
            msas=tuple(columns),
            row_pairing=np.stack(pairing_rows),
            row_is_paired=np.stack(paired_rows),
            target_instance_ids=tuple(target_instances),
        )
        return self._prepare_fold(structure, msa=collection)

    @staticmethod
    def _resize_msa_columns(msa, query_length: int, chain_id: str):
        """Trim surplus right-hand columns when RFO shortens chain A."""

        width = len(msa.unpaired_sequences[0])
        if width < query_length:
            raise ValueError(
                f"MSA for chain {chain_id!r} has query width {width}, "
                f"but the current sequence has length {query_length}"
            )
        if width == query_length:
            return msa
        return type(msa)(
            entity_id=msa.entity_id,
            unpaired_sequences=tuple(
                row[:query_length] for row in msa.unpaired_sequences
            ),
            unpaired_deletion_counts=msa.unpaired_deletion_counts[:, :query_length].copy(),
            paired_sequences=tuple(row[:query_length] for row in msa.paired_sequences),
            paired_deletion_counts=msa.paired_deletion_counts[:, :query_length].copy(),
        )

    def _sequence_probabilities(self, restype: torch.Tensor) -> torch.Tensor:
        class_indices = self._encoded_aa.to(device=restype.device)
        # RFO stores its sequence state as one-hot-like logits. A low softmax
        # temperature keeps the initial model input close to its hard sequence
        # while retaining gradients for alternative amino acids.
        return (restype.index_select(-1, class_indices) / 0.2).softmax(dim=-1).unsqueeze(0)

    def forward(
        self,
        *,
        trunk_input,
        confidence_input=None,
        pipeline_output=None,
        n_cycle: int | None = None,
        compute_gradient: bool = True,
        skip_diffusion: bool = False,
    ):
        if skip_diffusion:
            raise ValueError("The Protenix RFO backend currently requires confidence objectives")
        model_input = trunk_input["protenix_input"]
        sequence_probs = self._sequence_probabilities(trunk_input["f"]["restype"])
        if sequence_probs.shape != (*model_input.token_mask.shape, 20):
            raise ValueError(
                "RFO sequence probabilities and OFoldX Protenix tokens have incompatible "
                f"shapes: probabilities={tuple(sequence_probs.shape)}, "
                f"token_mask={tuple(model_input.token_mask.shape)}"
            )
        if not compute_gradient:
            with torch.no_grad():
                prediction = self.model.optimization_forward(
                    model_input,
                    sequence_probs,
                    num_recycles=n_cycle or self.n_recycles,
                    num_sampling_steps=self.num_sampling_steps,
                )
        else:
            prediction = self.model.optimization_forward(
                model_input,
                sequence_probs,
                num_recycles=n_cycle or self.n_recycles,
                num_sampling_steps=self.num_sampling_steps,
            )
        return {
            "pae": prediction.pae_logits,
            "pde": prediction.pde_logits,
            "plddt": prediction.plddt_logits,
            "distogram": prediction.distogram_logits,
            "coordinates": prediction.coordinates,
            "_prediction": prediction,
        }

    @staticmethod
    def _valid_token_pair_logits(model_output, name: str, pipeline_output):
        """Select real token pairs, excluding Protenix's padded batch slots."""
        logits = model_output[name]
        if logits.ndim != 5 or logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ValueError(
                f"Protenix {name} logits must have shape (1, 1, N, N, bins); "
                f"got {tuple(logits.shape)}"
            )
        indices = pipeline_output["token_indices"].to(device=logits.device)
        return logits[0, 0].index_select(0, indices).index_select(1, indices)

    @staticmethod
    def _plddt_per_token(model_output, pipeline_output):
        """Convert per-atom pLDDT logits into per-residue scores."""
        logits = model_output["plddt"]
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ValueError(
                "Protenix pLDDT logits must have shape (1, 1, A, bins); "
                f"got {tuple(logits.shape)}"
            )
        atom_indices = pipeline_output["atom_indices"].to(device=logits.device)
        atom_logits = logits[0, 0].index_select(0, atom_indices).float()
        centers = (torch.arange(atom_logits.shape[-1], device=logits.device) + 0.5) * (
            100.0 / atom_logits.shape[-1]
        )
        atom_scores = atom_logits.softmax(dim=-1) @ centers
        atom_to_token = pipeline_output["feats"]["atom_to_token_map"].to(
            device=logits.device, dtype=torch.long
        )
        token_count = len(pipeline_output["ground_truth"]["chain_iid_token_lvl"])
        if atom_to_token.numel() != atom_scores.numel():
            raise ValueError("Protenix pLDDT atoms and atom-to-token mapping have different lengths")
        if atom_to_token.numel() and (
            int(atom_to_token.min()) < 0 or int(atom_to_token.max()) >= token_count
        ):
            raise ValueError("Protenix atom-to-token mapping is outside the valid token range")
        totals = atom_scores.new_zeros((token_count,)).scatter_add_(0, atom_to_token, atom_scores)
        counts = atom_scores.new_zeros((token_count,)).scatter_add_(
            0, atom_to_token, torch.ones_like(atom_scores)
        )
        valid_tokens = counts > 0
        return totals / counts.clamp_min(1), valid_tokens

    def compute_loss(self, loss_type, trunk_input, model_output, confidence_input=None, pipeline_output=None):
        """Calculate RFO's configured score from Protenix confidence logits."""

        pae_logits = self._valid_token_pair_logits(model_output, "pae", pipeline_output).float()
        pae_centers = torch.linspace(
            0.0,
            32.0,
            pae_logits.shape[-1] + 1,
            device=pae_logits.device,
            dtype=pae_logits.dtype,
        )[:-1] + 16.0 / pae_logits.shape[-1]
        pae = pae_logits.softmax(dim=-1) @ pae_centers
        chain_labels = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        chain_ids, binder = self._chain_ids_and_binder_mask(chain_labels, pae.device)
        interface_mask = binder[:, None] ^ binder[None, :]
        interface_values = pae[interface_mask]

        if loss_type in ("pae_interface_mean", "pae_interface_min"):
            if interface_values.numel() == 0:
                return pae.new_tensor(float("inf"))
            return interface_values.mean() if loss_type == "pae_interface_mean" else interface_values.min()
        if loss_type == "pae_mean":
            return pae.mean()
        if loss_type == "pde_mean":
            logits = self._valid_token_pair_logits(model_output, "pde", pipeline_output).float()
            centers = torch.linspace(0.0, 32.0, logits.shape[-1] + 1, device=logits.device)[:-1]
            return (logits.softmax(dim=-1) @ (centers + 16.0 / logits.shape[-1])).mean()
        if loss_type == "plddt_mean":
            token_scores, valid_tokens = self._plddt_per_token(model_output, pipeline_output)
            if not bool(valid_tokens.any()):
                raise ValueError("Protenix produced no pLDDT scores for valid residues")
            return -token_scores[valid_tokens].mean()
        if loss_type == "iptm":
            cross_chain_mask = chain_ids[:, None] != chain_ids[None, :]
            if not bool(cross_chain_mask.any()):
                raise ValueError("The iPTM objective requires at least two chains.")
            effective_length = max(pae.shape[0], 19)
            d0 = 1.24 * (effective_length - 15.0) ** (1.0 / 3.0) - 1.8
            tm_like = 1.0 / (1.0 + (pae / d0).square())
            pair_counts = cross_chain_mask.sum(dim=-1)
            per_alignment = (tm_like * cross_chain_mask).sum(dim=-1) / pair_counts.clamp_min(1)
            # iPTM is a confidence to maximize, so RFO minimizes its negative.
            # Keep this calculation in Torch so gradients flow through PAE logits.
            return -per_alignment.max()
        raise ValueError(f"Unsupported Protenix objective: {loss_type}")

    @staticmethod
    def _chain_ids_and_binder_mask(chain_labels, device):
        """Return integer chain IDs and the RFO chain-A/design mask."""
        if isinstance(chain_labels, torch.Tensor):
            labels = chain_labels.to(device=device).reshape(-1)
            return labels, labels == 0

        import numpy as np

        labels = np.asarray(chain_labels).reshape(-1)
        if labels.dtype.kind in {"U", "S", "O"}:
            _, inverse = np.unique(labels, return_inverse=True)
            chain_ids = torch.as_tensor(inverse, device=device, dtype=torch.long)
            binder = torch.as_tensor(labels == "A", device=device)
            return chain_ids, binder
        chain_ids = torch.as_tensor(labels, device=device)
        return chain_ids, chain_ids == 0

    def optimization_objective(
        self,
        *,
        trunk_input,
        confidence_input=None,
        pipeline_output=None,
        objective_type="pae_interface_mean",
        n_cycle=None,
    ):
        output = self.forward(
            trunk_input=trunk_input,
            confidence_input=confidence_input,
            pipeline_output=pipeline_output,
            n_cycle=n_cycle,
        )
        return self.compute_loss(objective_type, trunk_input, output, confidence_input, pipeline_output)
