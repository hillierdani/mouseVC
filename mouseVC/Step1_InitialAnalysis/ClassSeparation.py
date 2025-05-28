import logging
import numpy as np
import scarf
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
from mouseVC import _cache_path, _load_or_run

logger = logging.getLogger(__name__)
from mouseVC.Step1_InitialAnalysis.definitions import cluster_dict, class_broad_map, leiden_markers, subclass_map


def apply_qc_filters_scarf(
		ds: scarf.DataStore,
		min_genes: int = 700,
		min_cells: int = 8,
		percent_mito_thresh: float = 0.01,
		max_genes: int = 6500,
		max_counts: int = 40000
) -> scarf.DataStore:
	"""Apply QC filters using Scarf's built-in methods"""

	# Filter cells and features using Scarf's efficient methods
	ds.filter_cells(
		min_features=min_genes,
		max_features=max_genes,
		max_mito_percent=percent_mito_thresh * 100,
		max_counts=max_counts
	)

	ds.filter_features(min_cells=min_cells)

	return ds


def import_10x_scarf(
		age_ids: List[str] = None,
		output_path: str = 'data/scarf_preHVG.zarr'
) -> scarf.DataStore:
	"""Simplified 10x import using AnnData intermediate step"""

	if age_ids is None:
		age_ids = ['P8', 'P14', 'P17', 'P21', 'P28', 'P38']

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_path.exists():
		return scarf.DataStore(str(output_path), nthreads=4)

	# Step 1: Use your existing AnnData concatenation logic
	# (copy from your working _compute_import_10x function)
	base = Path(__file__).parent.parent / 'ProcessedData' / 'Raw_Counts_Samples'
	default_suffs = ['_nr_1_a', '_nr_1_b', '_nr_2_a', '_nr_2_b']
	special = {'P38': ['_nr_1_a', '_nr_2_a', '_nr_2_b']}

	from scipy import sparse
	import scanpy as sc

	all_adatas = []

	for age in age_ids:
		suffs = special.get(age, default_suffs)
		age_reps = []

		for s in suffs:
			adata_temp = sc.read_10x_mtx(
				base / f"{age}{s}" / "filtered_feature_bc_matrix",
				var_names='gene_symbols',
				cache=True
			)
			adata_temp.var_names_make_unique()

			# Create unique cell names
			replicate_id = s.split('_nr_')[-1].replace('_', '')
			adata_temp.obs_names = [f"{age}_{replicate_id}_{bc}" for bc in adata_temp.obs_names]

			# Add metadata
			adata_temp.obs['batch'] = age
			adata_temp.obs['replicate'] = f"{age}_{replicate_id}"

			age_reps.append(adata_temp)

		# Concatenate replicates for this age
		if len(age_reps) > 1:
			age_adata = sc.concat(age_reps, merge='same')
		else:
			age_adata = age_reps[0]

		all_adatas.append(age_adata)

	# Concatenate all ages
	final_adata = sc.concat(all_adatas, merge='same')

	# Step 2: Save as h5ad temporarily
	temp_h5ad = output_path.with_suffix('.h5ad')
	final_adata.write_h5ad(temp_h5ad)

	# Step 3: Convert h5ad to Scarf using built-in converter
	reader = scarf.H5adReader(str(temp_h5ad))
	writer = scarf.H5adToZarr(
		reader,
		zarr_loc=str(output_path),
		chunk_size=(2000, 1000)
	)
	writer.dump()

	# Step 4: Clean up temporary file and create DataStore
	temp_h5ad.unlink()
	ds = scarf.DataStore(str(output_path), nthreads=4)

	# Apply QC filters
	ds = apply_qc_filters_scarf(ds)

	logger.info(f"Imported {ds.cells.N} cells with {ds.RNA.feats.N} features")
	return ds


def cluster_scarf(
		ds: scarf.DataStore,
		n_hvgs: int = 2000,
		n_pcs: int = 40,
		k_neighbors: int = 25,
		leiden_resolution: float = 0.5
) -> scarf.DataStore:
	"""Memory-efficient clustering using Scarf"""

	# HVG selection
	ds.mark_hvgs(
		min_cells=8,  # Minimum cells expressing the gene
		top_n=n_hvgs,  # Number of HVGs to select
		min_mean=0.0125,  # Minimum mean expression threshold
		max_mean=3,  # Maximum mean expression threshold
		max_var=6  # Maximum variance threshold
	)

	# Create neighborhood graph (includes normalization, PCA)
	ds.make_graph(
		feat_key='hvgs',
		dims=n_pcs,
		k=k_neighbors,
		n_centroids=100
	)

	# Leiden clustering
	ds.run_leiden_clustering(
		resolution=leiden_resolution,
		label='leiden_cluster'
	)

	# UMAP embedding
	ds.run_umap(
		feat_key='hvgs',
		n_epochs=200
	)

	logger.info(f"Clustering complete with {len(set(ds.cells.fetch('RNA_leiden_cluster')))} clusters")
	return ds


def doublet_detection_scarf(
		ds: scarf.DataStore,
		expected_doublet_rate: float = 0.06
) -> scarf.DataStore:
	"""Memory-efficient doublet detection using Scarf"""

	# Scarf has built-in doublet detection
	ds.mark_doublets(
		cluster_key='RNA_leiden_cluster',
		expected_doublet_rate=expected_doublet_rate
	)

	logger.info("Doublet detection complete")
	return ds


def annotate_clusters_scarf(
		ds: scarf.DataStore,
		cluster_map: Dict[int, str],
		class_broad_map: Dict[str, str]
) -> scarf.DataStore:
	"""Annotate clusters with names and broad classes using Scarf"""

	# Get leiden cluster assignments
	leiden_clusters = ds.cells.fetch('RNA_leiden_cluster')

	# Map to named clusters
	named_clusters = [cluster_map.get(int(cluster), f"Unknown_{cluster}")
					  for cluster in leiden_clusters]
	ds.cells.insert('cluster', named_clusters)

	# Map to broad classes
	broad_classes = [class_broad_map.get(str(cluster), 'Excitatory')
					 for cluster in leiden_clusters]
	ds.cells.insert('Class_broad', broad_classes)

	logger.info("Cluster annotation complete")
	return ds


def plot_comprehensive_analysis_scarf(
		ds: scarf.DataStore,
		output_dir: Path,
		leiden_markers: List[str] = None
) -> None:
	"""Generate comprehensive plots using Scarf's plotting functions"""

	output_dir.mkdir(parents=True, exist_ok=True)

	# Basic UMAP plots
	ds.plot_layout(
		layout_key='RNA_UMAP',
		color_by='RNA_leiden_cluster',
		save_as=str(output_dir / 'umap_leiden_clusters.pdf')
	)

	ds.plot_layout(
		layout_key='RNA_UMAP',
		color_by='Class_broad',
		save_as=str(output_dir / 'umap_broad_classes.pdf')
	)

	ds.plot_layout(
		layout_key='RNA_UMAP',
		color_by='cluster',
		save_as=str(output_dir / 'umap_named_clusters.pdf')
	)

	ds.plot_layout(
		layout_key='RNA_UMAP',
		color_by='batch',
		save_as=str(output_dir / 'umap_batch.pdf')
	)

	# Doublet visualization
	if 'is_doublet' in ds.cells.columns:
		ds.plot_layout(
			layout_key='RNA_UMAP',
			color_by='is_doublet',
			save_as=str(output_dir / 'umap_doublets.pdf')
		)

	# Cluster relationships
	ds.plot_cluster_tree(
		cluster_key='RNA_paris_cluster',
		save_as=str(output_dir / 'cluster_dendrogram.pdf')
	)

	# Feature plots for markers
	if leiden_markers:
		for i, marker in enumerate(leiden_markers[:10]):  # Top 10 markers
			if marker in ds.RNA.feats.fetch_all('names'):
				ds.plot_layout(
					layout_key='RNA_UMAP',
					color_by=marker,
					save_as=str(output_dir / f'umap_marker_{marker}.pdf')
				)

	logger.info(f"Plots saved to {output_dir}")


def plot_age_class_distributions_scarf(
		ds: scarf.DataStore,
		output_dir: Path
) -> None:
	"""Create age/class distribution plots using Scarf data"""

	# Extract metadata as DataFrame
	obs_df = pd.DataFrame({
		'batch': ds.cells.fetch('batch'),
		'Class_broad': ds.cells.fetch('Class_broad'),
		'cluster': ds.cells.fetch('cluster')
	})

	# Plot 1: Class distribution across ages
	class_order = ['Astrocytes', 'Oligodendrocytes', 'OPCs', 'Microglia',
				   'Endothelial', 'VLMC', 'Inhibitory', 'Excitatory', 'Ambiguous']

	ct1 = pd.crosstab(obs_df.batch, obs_df.Class_broad, normalize='index') * 100
	plot_class_order = [c for c in class_order if c in ct1.columns]
	ct1 = ct1.reindex(columns=plot_class_order, fill_value=0)

	fig, ax = plt.subplots(figsize=(20, 7))
	colors = sns.color_palette('Set1', n_colors=len(plot_class_order))
	ct1.plot(kind='bar', stacked=True, ax=ax, width=0.5, color=colors)
	ax.set_ylabel("Percentage of Cells per Age Group")
	ax.set_xlabel("Age Group (Batch)")
	plt.xticks(rotation=30, ha='right')
	plt.tight_layout()
	plt.savefig(output_dir / 'age_fraction_class.pdf')
	plt.close()

	# Plot 2: Age distribution across clusters
	ct2 = pd.crosstab(obs_df.cluster, obs_df.batch, normalize='index') * 100

	fig, ax = plt.subplots(figsize=(30, 8))
	age_colors = sns.color_palette('viridis', n_colors=len(ct2.columns))
	ct2.plot(kind='bar', stacked=True, ax=ax, width=0.5, color=age_colors)
	ax.set_ylabel("Percentage of Cells per Cluster")
	ax.set_xlabel("Cluster")
	plt.xticks(rotation=90, ha='right')
	plt.tight_layout()
	plt.savefig(output_dir / 'age_fraction_cluster.pdf')
	plt.close()


def split_by_class_and_age_scarf(
		ds: scarf.DataStore,
		output_dir: Path,
		neuron_classes: List[str] = ['Excitatory', 'Inhibitory']
) -> Dict[str, Path]:
	"""Split Scarf DataStore by class and age, saving as AnnData for compatibility"""

	output_dir.mkdir(parents=True, exist_ok=True)

	# Get all metadata
	obs_df = pd.DataFrame({
		'batch': ds.cells.fetch('batch'),
		'Class_broad': ds.cells.fetch('Class_broad'),
		'cluster': ds.cells.fetch('cluster')
	})

	saved_paths = {}

	# Save class-specific objects
	for class_name in neuron_classes + ['Non-neuron']:
		if class_name == 'Non-neuron':
			mask = ~obs_df['Class_broad'].isin(neuron_classes)
		else:
			mask = obs_df['Class_broad'] == class_name

		if mask.sum() > 0:
			# Get subset of cells
			subset_indices = np.where(mask)[0]

			# Create subset DataStore
			subset_path = output_dir / f'{class_name.lower()}_subset.zarr'
			ds_subset = ds.get_cell_subset(subset_indices, zarr_loc=str(subset_path))

			# Convert to AnnData for compatibility
			adata_path = output_dir / f'{class_name.lower()}.h5ad'
			convert_scarf_to_anndata(ds_subset, adata_path)
			saved_paths[class_name] = adata_path

	# Save age-specific objects for neuronal classes
	ages = obs_df['batch'].unique()

	for class_name in neuron_classes:
		class_mask = obs_df['Class_broad'] == class_name

		for age in ages:
			age_mask = obs_df['batch'] == age
			combined_mask = class_mask & age_mask

			if combined_mask.sum() > 0:
				subset_indices = np.where(combined_mask)[0]

				# Create age-specific subset
				subset_path = output_dir / f'{class_name.lower()}_{age}_subset.zarr'
				ds_subset = ds.get_cell_subset(subset_indices, zarr_loc=str(subset_path))

				# Convert to AnnData
				adata_path = output_dir / f'{class_name.lower()}_{age}.h5ad'
				convert_scarf_to_anndata(ds_subset, adata_path)
				saved_paths[f'{class_name}_{age}'] = adata_path

	return saved_paths


def convert_scarf_to_anndata(ds: scarf.DataStore, output_path: Path) -> None:
	"""Convert Scarf DataStore to AnnData for compatibility"""
	from anndata import AnnData
	import scipy.sparse as sp

	# Get expression matrix (use raw counts)
	X = ds.RNA.X

	# Get cell and feature names
	cell_names = ds.cells.fetch_all('names')
	feature_names = ds.RNA.feats.fetch_all('names')

	# Get metadata
	obs_data = {}
	for col in ds.cells.columns:
		if col != 'names':  # Skip the names column
			obs_data[col] = ds.cells.fetch(col)

	obs_df = pd.DataFrame(obs_data, index=cell_names)
	var_df = pd.DataFrame(index=feature_names)

	# Create AnnData
	adata = AnnData(X=X, obs=obs_df, var=var_df)

	# Add UMAP if available
	if 'RNA_UMAP_1' in ds.cells.columns:
		adata.obsm['X_umap'] = np.column_stack([
			ds.cells.fetch('RNA_UMAP_1'),
			ds.cells.fetch('RNA_UMAP_2')
		])

	# Save
	adata.write_h5ad(output_path)
	logger.info(f"Converted Scarf DataStore to AnnData: {output_path}")


def main_scarf(age: List[str] = None, max_memory_gb: float = 16) -> None:
	"""Main Scarf-based analysis pipeline"""

	if age is None:
		age = ['P8', 'P28']

	age = [a.upper() for a in age]
	age_tag = '_'.join(age)

	# Setup directories
	data_dir = Path('data') / age_tag
	fig_dir = Path('figures') / age_tag
	data_dir.mkdir(parents=True, exist_ok=True)
	fig_dir.mkdir(parents=True, exist_ok=True)

	logger.info(f"Starting Scarf-based analysis for {age_tag}")
	logger.info(f"Target memory usage: <{max_memory_gb}GB")

	# Step 1: Import data using Scarf
	zarr_path = data_dir / f'scarf_data_{age_tag}.zarr'
	ds = import_10x_scarf(age, str(zarr_path))

	# Step 2: Clustering pipeline
	ds = cluster_scarf(ds)

	# Step 3: Doublet detection
	ds = doublet_detection_scarf(ds)

	# Step 4: Annotate clusters
	ds = annotate_clusters_scarf(ds, cluster_dict, class_broad_map)

	# Step 5: Generate comprehensive plots
	plot_comprehensive_analysis_scarf(ds, fig_dir, leiden_markers)
	plot_age_class_distributions_scarf(ds, fig_dir)

	# Step 6: Remove doublets and ambiguous cells
	doublet_mask = ds.cells.fetch('is_doublet') if 'is_doublet' in ds.cells.columns else [False] * ds.cells.N
	cluster_names = ds.cells.fetch('cluster')
	ambiguous_mask = [name.startswith('Ambig_') for name in cluster_names]

	# Create clean dataset
	clean_mask = [not (d or a) for d, a in zip(doublet_mask, ambiguous_mask)]
	clean_indices = np.where(clean_mask)[0]

	if len(clean_indices) > 0:
		clean_zarr_path = data_dir / f'scarf_clean_{age_tag}.zarr'
		ds_clean = ds.get_cell_subset(clean_indices, zarr_loc=str(clean_zarr_path))

		# Step 7: Split by class and age
		split_paths = split_by_class_and_age_scarf(
			ds_clean,
			data_dir / 'split_objects'
		)

		logger.info(f"Analysis complete. Split objects saved:")
		for name, path in split_paths.items():
			logger.info(f"  {name}: {path}")

	# Step 8: Save final DataStore location
	final_info_path = data_dir / 'scarf_analysis_info.txt'
	with open(final_info_path, 'w') as f:
		f.write(f"Scarf analysis complete for {age_tag}\n")
		f.write(f"Main DataStore: {zarr_path}\n")
		f.write(f"Clean DataStore: {clean_zarr_path}\n")
		f.write(f"Figures: {fig_dir}\n")
		f.write(f"Split objects: {data_dir / 'split_objects'}\n")

	logger.info(f"Scarf-based analysis complete for {age_tag}")


if __name__ == '__main__':
	import sys
	import fire

	if len(sys.argv) == 1:
		main_scarf()
	else:
		fire.Fire({
			'main': main_scarf,
			'import_only': import_10x_scarf,
			'cluster_only': cluster_scarf
		})
