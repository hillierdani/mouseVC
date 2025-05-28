#!/usr/bin/env python
"""
Generate L23_mapped files for any age combination

Maps L2/3 cells from any query age to reference age annotations using correlation-based mapping.
"""

import logging
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure scanpy
sc.settings.verbosity = 2
sc.set_figure_params(dpi=100)


class L23ReferenceMapper:
	"""
	Maps L2/3 cells from any query age to any reference age annotations.
	"""

	def __init__(self):
		self.l23_marker_genes = [
			'Cux2', 'Ccbe1', 'Mdga1', 'Cdh13', 'Trpc6', 'Chrm2',
			'Robo1', 'Dpyd', 'Col23a1', 'Rhbdl3', 'Adamts2',
			'Epha6', 'Epha3', 'Dscaml1', 'Grm1', 'Igfn1', 'Bdnf',
			'Nell1', 'Cdh12', 'Kcnh5', 'Ncam2', 'Pld5', 'Rorb',
			'Tox', 'Parm1', 'Cntn5', 'Astn2'
		]

		# Available developmental ages
		self.available_ages = ['P8', 'P14', 'P28', 'P56']

	def validate_ages(self, query_age: str, reference_age: str) -> None:
		"""
		Validate that provided ages are available.

		Args:
			query_age: Age of cells to be mapped
			reference_age: Reference age for mapping
		"""
		if query_age not in self.available_ages:
			raise ValueError(f"Query age {query_age} not in available ages: {self.available_ages}")

		if reference_age not in self.available_ages:
			raise ValueError(f"Reference age {reference_age} not in available ages: {self.available_ages}")

		if query_age == reference_age:
			logger.warning(f"Query and reference are the same age ({query_age}). Mapping may not be meaningful.")

	def load_datasets(
			self,
			query_path: str,
			reference_path: str
	) -> Tuple[sc.AnnData, sc.AnnData]:
		"""
		Load query and reference glutamatergic datasets.

		Args:
			query_path: Path to query glutamatergic dataset
			reference_path: Path to reference glutamatergic dataset

		Returns:
			Tuple of (query_data, reference_data)
		"""
		logger.info("Loading datasets...")

		# Load query data
		if Path(query_path).exists():
			query_data = sc.read_h5ad(query_path)
		else:
			raise FileNotFoundError(f"Query file not found: {query_path}")

		# Load reference data
		if Path(reference_path).exists():
			reference_data = sc.read_h5ad(reference_path)
		else:
			raise FileNotFoundError(f"Reference file not found: {reference_path}")

		logger.info(f"Loaded query: {query_data.n_obs} cells, reference: {reference_data.n_obs} cells")
		return query_data, reference_data

	def extract_l23_cells(self, adata: sc.AnnData, age: str) -> sc.AnnData:
		"""
		Extract L2/3 cells from dataset.

		Args:
			adata: Input AnnData object
			age: Age label for logging

		Returns:
			L2/3 subset of the data
		"""
		# Extract L2/3 cells
		l23_mask = adata.obs['Subclass'] == 'L2/3'
		l23_data = adata[l23_mask, :].copy()

		logger.info(f"Extracted {l23_data.n_obs} L2/3 cells from {age}")
		return l23_data

	def prepare_reference_profiles(self, reference_l23: sc.AnnData, reference_age: str) -> Tuple[
		Dict[str, np.ndarray], List[str]]:
		"""
		Create reference expression profiles for each reference L2/3 cell type.

		Args:
			reference_l23: Reference L2/3 AnnData object
			reference_age: Reference age label for logging

		Returns:
			Tuple of (reference_profiles dict, available_markers list)
		"""
		logger.info(f"Computing {reference_age} L2/3 reference profiles...")

		# Get available marker genes
		available_markers = [gene for gene in self.l23_marker_genes
							 if gene in reference_l23.var_names]
		logger.info(f"Using {len(available_markers)} marker genes for mapping")

		# Subset to marker genes
		ref_markers = reference_l23[:, available_markers].copy()

		# Use raw counts if available, otherwise use X
		if hasattr(ref_markers, 'raw') and ref_markers.raw is not None:
			expr_matrix = ref_markers.raw.X
		else:
			expr_matrix = ref_markers.X

		# Convert to dense if sparse
		if hasattr(expr_matrix, 'toarray'):
			expr_matrix = expr_matrix.toarray()

		# Create reference profiles for each cell type
		reference_profiles = {}
		cell_types = ref_markers.obs['Type'].cat.categories

		for cell_type in cell_types:
			if not cell_type.startswith('L2/3'):
				continue

			mask = ref_markers.obs['Type'] == cell_type
			if mask.sum() == 0:
				continue

			# Compute mean expression profile
			type_cells = expr_matrix[mask, :]
			mean_profile = np.mean(type_cells, axis=0)
			reference_profiles[cell_type] = mean_profile

			logger.info(f"Reference {cell_type}: {mask.sum()} cells")

		return reference_profiles, available_markers

	def map_cells_to_reference(
			self,
			query_l23: sc.AnnData,
			reference_profiles: Dict[str, np.ndarray],
			marker_genes: List[str],
			query_age: str,
			reference_age: str,
			confidence_threshold: float = 0.3
	) -> sc.AnnData:
		"""
		Map query L2/3 cells to reference cell types.

		Args:
			query_l23: Query L2/3 cells
			reference_profiles: Reference profiles
			marker_genes: List of marker genes used for mapping
			query_age: Query age label
			reference_age: Reference age label
			confidence_threshold: Minimum correlation for confident mapping

		Returns:
			Query L2/3 cells with reference type annotations
		"""
		logger.info(f"Mapping {query_age} L2/3 cells to {reference_age} reference...")

		# Subset to marker genes
		query_markers = query_l23[:, marker_genes].copy()

		# Get expression matrix
		if hasattr(query_markers, 'raw') and query_markers.raw is not None:
			query_expr = query_markers.raw.X
		else:
			query_expr = query_markers.X

		# Convert to dense if sparse
		if hasattr(query_expr, 'toarray'):
			query_expr = query_expr.toarray()

		# Map each query cell to best matching reference cell type
		ref_assignments = []
		mapping_scores = []

		for i in range(query_expr.shape[0]):
			cell_expr = query_expr[i, :]

			# Compute correlation with each reference profile
			correlations = {}
			for cell_type, ref_profile in reference_profiles.items():
				# Use Pearson correlation
				if np.std(cell_expr) > 0 and np.std(ref_profile) > 0:
					corr, _ = pearsonr(cell_expr, ref_profile)
					correlations[cell_type] = corr if not np.isnan(corr) else 0
				else:
					correlations[cell_type] = 0

			# Find best match
			if correlations:
				best_type = max(correlations, key=correlations.get)
				best_score = correlations[best_type]

				# Apply confidence threshold
				if best_score >= confidence_threshold:
					ref_assignments.append(best_type)
					mapping_scores.append(best_score)
				else:
					ref_assignments.append('Unassigned')
					mapping_scores.append(best_score)
			else:
				ref_assignments.append('Unassigned')
				mapping_scores.append(0.0)

		# Add annotations to query data
		query_mapped = query_l23.copy()
		query_mapped.obs[f'{reference_age}_Type'] = pd.Categorical(ref_assignments)
		query_mapped.obs['mapping_score'] = mapping_scores

		# Count assignments
		assignment_counts = pd.Series(ref_assignments).value_counts()
		logger.info("Mapping results:")
		for cell_type, count in assignment_counts.items():
			logger.info(f"  {cell_type}: {count} cells")

		return query_mapped

	def create_mapped_file(
			self,
			query_age: str = 'P8',
			reference_age: str = 'P28',
			query_path: str = None,
			reference_path: str = None,
			output_path: str = None,
			confidence_threshold: float = 0.3
	) -> sc.AnnData:
		"""
		Complete pipeline to create the mapped file.

		Args:
			query_age: Age of cells to be mapped
			reference_age: Reference age for mapping
			query_path: Path to query dataset (auto-generated if None)
			reference_path: Path to reference dataset (auto-generated if None)
			output_path: Output path for mapped file (auto-generated if None)
			confidence_threshold: Minimum correlation for confident mapping

		Returns:
			Mapped query L2/3 AnnData object
		"""
		# Validate ages
		self.validate_ages(query_age, reference_age)

		# Auto-generate paths if not provided
		if query_path is None:
			query_path = f'{query_age}_glut_analyzed.h5ad'

		if reference_path is None:
			reference_path = f'{reference_age}_glut_analyzed.h5ad'

		if output_path is None:
			output_path = f'{query_age}_L23_mapped_to_{reference_age}_L23.h5ad'

		try:
			# Load datasets
			query_data, reference_data = self.load_datasets(query_path, reference_path)

			# Extract L2/3 cells
			query_l23 = self.extract_l23_cells(query_data, query_age)
			reference_l23 = self.extract_l23_cells(reference_data, reference_age)

			# Create reference profiles
			reference_profiles, marker_genes = self.prepare_reference_profiles(reference_l23, reference_age)

			# Map query cells to reference
			query_mapped = self.map_cells_to_reference(
				query_l23, reference_profiles, marker_genes,
				query_age, reference_age, confidence_threshold
			)

			# Save mapped file
			query_mapped.write_h5ad(output_path)
			logger.info(f"Saved mapped file to {output_path}")

			# Generate summary plots
			self.plot_mapping_summary(query_mapped, output_path, query_age, reference_age)

			return query_mapped

		except Exception as e:
			logger.error(f"Mapping failed: {e}")
			raise

	def plot_mapping_summary(self, adata: sc.AnnData, output_path: str, query_age: str, reference_age: str) -> None:
		"""
		Generate summary plots for the mapping results.

		Args:
			adata: Mapped AnnData object
			output_path: Base output path for saving plots
			query_age: Query age label
			reference_age: Reference age label
		"""
		import matplotlib.pyplot as plt

		try:
			# Plot mapping results
			fig, axs = plt.subplots(1, 3, figsize=(15, 4))

			# Original query types
			sc.pl.umap(adata, color='Type', ax=axs[0], show=False,
					   title=f'Original {query_age} Types', legend_loc='on data')

			# Mapped reference types
			ref_type_col = f'{reference_age}_Type'
			sc.pl.umap(adata, color=ref_type_col, ax=axs[1], show=False,
					   title=f'Mapped {reference_age} Types', legend_loc='on data')

			# Mapping confidence scores
			sc.pl.umap(adata, color='mapping_score', ax=axs[2], show=False,
					   title='Mapping Confidence', cmap='viridis')

			plt.tight_layout()
			plot_path = output_path.replace('.h5ad', '_mapping_summary.pdf')
			plt.savefig(plot_path)
			plt.close()

			logger.info(f"Saved mapping summary plot to {plot_path}")

		except Exception as e:
			logger.warning(f"Could not generate plots: {e}")


def map_age_to_reference(
		query_age: str = 'P8',
		reference_age: str = 'P28',
		confidence_threshold: float = 0.3
):
	"""
	Map any age to any reference age.

	Args:
		query_age: Age to map from (P8, P14, P28, P56)
		reference_age: Reference age to map to (P8, P14, P28, P56)
		confidence_threshold: Minimum correlation threshold
	"""
	mapper = L23ReferenceMapper()

	try:
		mapped_data = mapper.create_mapped_file(
			query_age=query_age,
			reference_age=reference_age,
			confidence_threshold=confidence_threshold
		)

		print(f"Successfully created {query_age} -> {reference_age} mapped file!")
		print(f"Total {query_age} L2/3 cells: {mapped_data.n_obs}")
		print(f"{reference_age} type assignments:")
		ref_type_col = f'{reference_age}_Type'
		print(mapped_data.obs[ref_type_col].value_counts())
		print(f"Mean mapping score: {mapped_data.obs['mapping_score'].mean():.3f}")

	except FileNotFoundError as e:
		print(f"Required input files not found: {e}")
		print("Please ensure you have:")
		print(f"1. {query_age}_glut_analyzed.h5ad")
		print(f"2. {reference_age}_glut_analyzed.h5ad")
	except Exception as e:
		print(f"Error creating mapped file: {e}")


def map_all_to_reference(reference_age: str = 'P28'):
	"""
	Map all available ages to a reference age.

	Args:
		reference_age: Reference age for all mappings
	"""
	mapper = L23ReferenceMapper()
	available_ages = ['P8', 'P14', 'P28', 'P56']

	for query_age in available_ages:
		if query_age == reference_age:
			continue

		print(f"\n=== Mapping {query_age} to {reference_age} ===")
		try:
			map_age_to_reference(query_age, reference_age)
		except Exception as e:
			print(f"Failed to map {query_age} to {reference_age}: {e}")


def main(
		query_age: str = 'P8',
		reference_age: str = 'P28',
		confidence_threshold: float = 0.3,
		map_all: bool = False
):
	"""
	Main execution function with flexible age mapping.

	Args:
		query_age: Age to map from
		reference_age: Reference age to map to
		confidence_threshold: Minimum correlation threshold
		map_all: If True, map all ages to reference_age
	"""
	if map_all:
		map_all_to_reference(reference_age)
	else:
		map_age_to_reference(query_age, reference_age, confidence_threshold)


if __name__ == '__main__':
	import fire

	fire.Fire(main)
