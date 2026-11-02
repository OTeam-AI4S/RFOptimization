import logging
import tempfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

from rfo.backprop.inference_pipeline.protenix_inference_pipeline import ProtenixInferencePipeline
from rfo.backprop.tools.af3 import decode_restype, seq2cif
from rfo.backprop.tools.mutate import compare_seq_mutation, mutate_reslogits
from rfo.backprop.tools.utils import norm_seq_grad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelhubGradient:
    """
    Protenix gradient-based sequence optimizer using OFoldX for model interactions.
    """

    def __init__(self, seqopt_config=None):
        """Initialize the optimizer with the configured OFoldX Protenix model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seqopt_config = seqopt_config
        model_name = getattr(seqopt_config, "protenix_model", None)
        if not model_name:
            raise ValueError(
                "Set protenix_model to an OFoldX Protenix model repository or local directory"
            )
        self.inference_pipeline = ProtenixInferencePipeline(
            str(model_name),
            device=self.device,
            n_recycles=getattr(seqopt_config, "n_recycles", 10),
            num_sampling_steps=getattr(seqopt_config, "num_steps", None),
        )
        # GradMCMC supports checkpoint ensembles through this interface. The
        # standalone Protenix backend currently uses one configured pipeline.
        self.inference_pipelines = [self.inference_pipeline]
        self.n_models = len(self.inference_pipelines)
        self.pipeline = self.inference_pipeline.pipeline
        self.encoding = self.inference_pipeline.encoding
        self.msa_paths = getattr(seqopt_config, "msa_paths", None)
        self.paired_msa_paths = getattr(seqopt_config, "paired_msa_paths", None)
        self.cif_add_info = None
        logger.info("Optimizer initialized with OFoldX Protenix")

    def project_with_cif_reconstruction(
        self, seq, cif_add_info, protein_mask, out_file=None
    ):
        """
        Takes a protein sequence as input, writes it to a CIF file, and reads back
        via pipeline transformation for consistency in model features.
        """
        with tempfile.NamedTemporaryFile(delete=True, suffix=".cif.gz") as tmp_file:
            cif_path, components = seq2cif(
                seq,
                add_info=cif_add_info,
                protein_mask=protein_mask,
                out_file=out_file or tmp_file.name,
            )
            model_input = self.inference_pipeline.prepare_fold_with_msa(
                cif_path,
                (cif_add_info or {}).get("msa_paths"),
                (cif_add_info or {}).get("paired_msa_paths"),
            )
        return (
            self._protenix_rfo_inputs(model_input),
            {},
            self._protenix_rfo_output(model_input),
            components,
        )

    def prepare_initial_structure(
        self,
        structure_path,
        *,
        fix_template_dict=None,
        self_template_selection_syntax=None,
    ):
        """Prepare the Protenix path's initial structure and RFO sequence tensors."""

        self.inference_pipeline.configure_templates(
            structure_path,
            fix_template_dict=(
                fix_template_dict
                if fix_template_dict is not None
                else getattr(self.seqopt_config, "fix_template_dict", None)
            ),
            self_template_selection_syntax=(
                self_template_selection_syntax
                if self_template_selection_syntax is not None
                else getattr(self.seqopt_config, "self_template_selection_syntax", None)
            ),
        )
        model_input = self.inference_pipeline.prepare_fold_with_msa(
            structure_path, self.msa_paths, self.paired_msa_paths
        )
        return (
            self._protenix_rfo_inputs(model_input),
            {},
            self._protenix_rfo_output(model_input),
            None,
        )

    def _protenix_rfo_inputs(self, model_input):
        from ofoldx.data.tokenization import TOKEN_INDEX
        from atomworks.constants import DICT_THREE_TO_ONE

        model_features = model_input.features
        token_features = model_features.token_features[0]
        token_mask = model_features.token_mask[0].bool()
        protein_mask = model_features.token_has_esm[0].bool() & token_mask
        valid_tokens = token_mask.nonzero(as_tuple=False).flatten()
        valid_proteins = protein_mask.nonzero(as_tuple=False).flatten()
        vocab = token_features[valid_tokens, 384 : 384 + len(TOKEN_INDEX)]
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        protein_tokens = model_features.token_has_esm[0, valid_tokens].tolist()
        sequence = "".join(
            next((aa for aa in aa_order if TOKEN_INDEX[aa] == index), "?")
            for index, is_protein in zip(vocab.argmax(dim=-1).tolist(), protein_tokens)
            if is_protein
        )
        if "?" in sequence:
            raise ValueError(
                "RFO's Protenix sequence optimizer supports canonical amino acids only; "
                "the input contains a protein token outside ACDEFGHIKLMNPQRSTVWY."
            )
        if len(sequence) != valid_proteins.numel():
            raise ValueError("Protenix protein-token sequence and residue mask have different lengths")
        aa_to_three = {one: three for three, one in DICT_THREE_TO_ONE.items()}
        encoded = torch.as_tensor(
            [int(self.encoding.encode(aa_to_three[amino_acid])) for amino_acid in sequence],
            dtype=torch.long,
            device=self.device,
        )
        restype = torch.zeros(
            (token_mask.numel(), 32), dtype=torch.float32, device=self.device
        )
        if encoded.numel():
            restype[valid_proteins] = torch.nn.functional.one_hot(
                encoded, num_classes=32
            ).float()
        return {
            "protenix_input": model_features,
            "f": {"restype": restype, "is_protein": protein_mask},
        }

    @staticmethod
    def _protenix_rfo_output(model_input):
        model_features = model_input.features
        system = model_input.context.structure.system
        token_indices = model_features.token_mask[0].bool().nonzero(as_tuple=False).flatten()
        atom_indices = model_features.atom_mask[0].bool().nonzero(as_tuple=False).flatten()
        token_chain_indices = model_features.chain_index[0, token_indices].detach().cpu().tolist()
        chain_labels = np.asarray(
            [system.text("instance", system.instance_symbol[index : index + 1])[0] for index in token_chain_indices]
        )
        compact_token_indices = torch.full_like(model_features.token_mask[0], -1, dtype=torch.long)
        compact_token_indices[token_indices] = torch.arange(token_indices.numel(), device=token_indices.device)
        atom_to_token = compact_token_indices[
            model_features.atom_to_token[0, atom_indices].long()
        ]
        if bool((atom_to_token < 0).any()):
            raise ValueError("Protenix maps a valid atom to an invalid token")
        return {
            "protenix_input": model_features,
            "prepared_input": model_input,
            "ground_truth": {
                "chain_iid_token_lvl": chain_labels,
                "coord_atom_lvl": model_features.reference_positions[0, atom_indices].detach().cpu(),
            },
            "feats": {"atom_to_token_map": atom_to_token.detach().cpu()},
            "token_indices": token_indices,
            "atom_indices": atom_indices,
            "protein_token_mask": model_features.token_has_esm[0, token_indices].bool(),
        }

    def my_objective(self, trunk_input, confidence_input, pipeline_output=None):
        """
        Compute objective using inference pipeline.

        The objective name comes from ``loss_type`` in the optimizer config.
        """
        objective_type = getattr(self.seqopt_config, "loss_type", "pae_interface_mean")

        return self.inference_pipeline.optimization_objective(
            trunk_input=trunk_input,
            confidence_input=confidence_input,
            pipeline_output=pipeline_output,
            objective_type=objective_type,
            n_cycle=self.inference_pipeline.n_recycles,
        )

    @staticmethod
    def extract_info_from_model_output(
        inference_pipeline, model_output, pipeline_output, expect_full_outputs=True
    ):
        """
        Extract relevant information from the pipeline output.

        Args:
            inference_pipeline: The inference pipeline instance
            model_output: Output from the model forward pass
            pipeline_output: Pipeline output dictionary
            expect_full_outputs: Whether to expect full outputs (False when skip_diffusion=True)

        Returns:
            Dictionary with extracted information.
        """
        results = {}
        prediction = model_output.get("_prediction")
        if prediction is None:
            if expect_full_outputs:
                results.update(
                    atom_array_stack=None,
                    X_L=None,
                    ipae_mean=float("inf"),
                    ipae_min=float("inf"),
                )
            return results

        from ofoldx.models.protenix.pipeline import ProtenixPipelineInput

        prepared_input = pipeline_output["prepared_input"]
        results["X_L"] = prediction.coordinates.detach()
        # Structure decoding and reported metrics are outside autograd; the
        # separately computed objective retains its gradient graph.
        with torch.no_grad():
            decoded = inference_pipeline.pipeline.postprocess(
                ProtenixPipelineInput(prepared_input, None), prediction
            )
            results["atom_array_stack"] = decoded.predictions[0].structure
            results["ipae_mean"] = float(
                inference_pipeline.compute_loss(
                    "pae_interface_mean", {}, model_output, {}, pipeline_output
                ).cpu()
            )
            results["ipae_min"] = float(
                inference_pipeline.compute_loss(
                    "pae_interface_min", {}, model_output, {}, pipeline_output
                ).cpu()
            )
            results["iptm"] = float(
                -inference_pipeline.compute_loss(
                    "iptm", {}, model_output, {}, pipeline_output
                ).cpu()
            )
            results["plddt_mean"] = float(
                -inference_pipeline.compute_loss(
                    "plddt_mean", {}, model_output, {}, pipeline_output
                ).cpu()
            )

            plddt_per_residue, valid_residues = inference_pipeline._plddt_per_token(
                model_output, pipeline_output
            )
            chain_labels = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
            _chain_ids, binder_mask = inference_pipeline._chain_ids_and_binder_mask(
                chain_labels, plddt_per_residue.device
            )
            protein_mask = pipeline_output["protein_token_mask"].to(
                device=plddt_per_residue.device, dtype=torch.bool
            )
            substrate_mask = (~binder_mask) & protein_mask & valid_residues
            results["substrate_plddt"] = (
                float(plddt_per_residue[substrate_mask].mean().cpu())
                if bool(substrate_mask.any())
                else float("nan")
            )
        return results

    def inference_without_gradient(
        self, trunk_input, confidence_input, pipeline_output=None
    ):
        trunk_output = self.inference_pipeline.forward(
            trunk_input=trunk_input,
            confidence_input=confidence_input,
            pipeline_output=pipeline_output,
            n_cycle=self.inference_pipeline.n_recycles,
            compute_gradient=False,
            skip_diffusion=False,
        )

        additional_info = self.extract_info_from_model_output(
            self.inference_pipeline,
            trunk_output,
            pipeline_output,
            expect_full_outputs=True,
        )

        return additional_info

    def save_structure(self, atom_array_stack, output_path):
        """
        Save atom array stack as CIF file.

        Args:
            atom_array_stack: AtomArrayStack to save
            output_path: Path for output file (without extension)
        """
        from ofoldx.io import write_structure

        write_structure(atom_array_stack, output_path)

    def grad_descent(self, verbose=True):
        """Perform gradient descent on the restype tensor."""
        self.optimizer.zero_grad()
        self.trunk_input["f"]["restype"] = self.restype

        # get loss with gradients from inference pipeline
        loss = self.my_objective(
            self.trunk_input, self.confidence_input, self.pipeline_output
        )
        loss.backward()

        # normalize the gradient
        with torch.no_grad():
            new_grad = norm_seq_grad(self.restype.grad)
            new_grad = new_grad * self.seq_of_interest_mask.unsqueeze(-1).float()
            self.restype.grad.copy_(new_grad)

        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        self.history.append(
            {
                "step": self.step,
                "loss": float(loss.cpu().detach().numpy()),
                "sequence": decode_restype(
                    self.restype.cpu().detach(), self.encoding, self.protein_mask
                ),
            }
        )

        if verbose:
            print(f"[Step {self.step:3d}] loss={loss.item():.4f}")

    def seq2loss(self, seq, out_file=None, protein_mask=None, cif_add_info=None):
        """Evaluate sequence loss using inference pipeline."""
        # Use provided values or fall back to instance attributes
        protein_mask = (
            protein_mask
            if protein_mask is not None
            else getattr(self, "protein_mask", None)
        )
        cif_add_info = (
            cif_add_info
            if cif_add_info is not None
            else getattr(self, "cif_add_info", None)
        )

        if protein_mask is None:
            raise ValueError(
                "protein_mask must be provided or set as an instance attribute"
            )

        with torch.no_grad():
            new_trunk_input, new_confidence_input, new_pipeline_output, _ = (
                self.project_with_cif_reconstruction(
                    seq=seq,
                    cif_add_info=cif_add_info,
                    protein_mask=protein_mask,
                    out_file=out_file,
                )
            )

            loss = self.my_objective(
                new_trunk_input, new_confidence_input, new_pipeline_output
            )

        return loss

    def sequence_gradient_opt(
        self,
        init_restype: torch.Tensor,
        trunk_input: dict,
        confidence_input: dict,
        pipeline_output: dict,
        mutate_sample_step: int = 1,
        opt_steps: int = 100,
        lr: float = 1e-2,
        round_trip: bool = True,
        seq_of_interest_mask=None,
        random_mut: bool = False,
        max_mut=None,
        pos_mut_prob=None,
        cif_add_info=None,
    ):
        """Perform gradient-based sequence optimization."""
        # initialize trainable restype parameter
        self.init_restype = init_restype
        self.restype = torch.nn.Parameter(
            init_restype.clone().detach().float().to(self.device)
        )
        self.restype.requires_grad = True

        self.optimizer = torch.optim.Adam([self.restype], lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)
        self.trunk_input = trunk_input
        self.confidence_input = confidence_input
        self.pipeline_output = pipeline_output
        self.protein_mask = trunk_input["f"]["is_protein"]
        self.cif_add_info = cif_add_info
        self.history = []

        results = {"evolve_traj": []}

        # Initialize sequence tracking
        self.initial_seq_str = decode_restype(
            init_restype, self.encoding, self.protein_mask
        )
        previous_seq_str = self.initial_seq_str
        print("Initial sequence:", previous_seq_str)

        self.seq_of_interest_mask = seq_of_interest_mask
        if seq_of_interest_mask is not None:
            seq_of_interest_index = torch.where(seq_of_interest_mask)[0].cpu().numpy()
            print("################################################################")
            print(f"Positions allowed to mutate: {seq_of_interest_index}.")
            print("################################################################")

        self.step = 0
        for step in range(opt_steps):
            self.grad_descent()

            # apply mutations periodically
            if (
                (mutate_sample_step > 0)
                and (step % mutate_sample_step == 0)
                and (step > 0)
            ):
                mutated_restype, mutated_seq = mutate_reslogits(
                    old_seq_str=previous_seq_str,
                    res_logits=self.restype,
                    encoding=self.encoding,
                    random_mut=random_mut,
                    allowed_mutate_mask=seq_of_interest_mask,
                    max_mut=max_mut,
                    pos_mut_prob=pos_mut_prob,
                    prot_dim=20,
                    full_dim=32,
                    force_mutate=False,
                )

                mutation_positions = compare_seq_mutation(
                    previous_seq_str, mutated_seq, print_num=5
                )

                # update restype
                with torch.no_grad():
                    self.restype.copy_(mutated_restype)

                if round_trip and (len(mutation_positions) > 0):
                    new_trunk_input, new_confidence_input, new_pipeline_output, _ = (
                        self.project_with_cif_reconstruction(
                            seq=mutated_seq,
                            cif_add_info=cif_add_info,
                            protein_mask=self.protein_mask,
                        )
                    )
                    self.trunk_input = new_trunk_input
                    self.confidence_input = new_confidence_input
                    self.pipeline_output = new_pipeline_output

                    print("Performing round trip transformation...")

                    with torch.no_grad():
                        self.restype.copy_(new_trunk_input["f"]["restype"])
                        real_loss = self.my_objective(
                            new_trunk_input, new_confidence_input, self.pipeline_output
                        )

                        results["evolve_traj"].append(
                            {
                                "step": step,
                                "loss": real_loss.item(),
                                "sequence": mutated_seq,
                            }
                        )

                    print(f"Real Loss: [Step {step:3d}] loss={real_loss.item():.4f}")

                previous_seq_str = mutated_seq

        print("\nOptimization complete!")
        return self.history, results
