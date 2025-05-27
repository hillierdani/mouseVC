import logging
import numpy as np
import scanpy as sc
from pathlib import Path
from anndata import AnnData
import anndata
import scrublet
import pandas as pd
import seaborn as sns # For fallback color palettes
import matplotlib.pyplot as plt


from mouseVC import _cache_path, _load_or_run

logger = logging.getLogger(__name__)
from mouseVC.Step1_InitialAnalysis.definitions import cluster_dict, class_broad_map, leiden_markers, subclass_map

def apply_qc_filters(
    adata: AnnData,
    min_genes: int = 700,
    min_cells: int = 8,
    percent_mito_thresh: float = 0.01,
    max_genes: int = 6500,
    max_counts: int = 40000
) -> AnnData:
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    mito_mask = adata.var_names.str.startswith('mt-')
    adata.obs['percent_mito'] = (
        np.sum(adata[:, mito_mask].X, axis=1).A1 /
        np.sum(adata.X, axis=1).A1
    )
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    return adata[
        (adata.obs.percent_mito < percent_mito_thresh) &
        (adata.obs.n_genes < max_genes) &
        (adata.obs.n_counts < max_counts),
    ]

def import_10x(age_ids=None):
    age_ids = age_ids or ['P8','P14','P17','P21','P28','P38']
    cache_file = _cache_path('preHVG', *age_ids)
    return _load_or_run(cache_file, lambda: _compute_import_10x(age_ids))

def _compute_import_10x(age_ids=None):
    """
    Read raw 10x matrices per age, concatenate without filtering,
    compute QC metrics and plot before filtering,
    apply QC filters, plot after filtering,
    normalize, log-transform, set raw, and save to `PooledMVC_preHVG.h5ad`.
    """
    if age_ids is None:
        age_ids = ['P8', 'P14', 'P17', 'P21', 'P28', 'P38']
    base = Path(__file__).parent.parent / 'ProcessedData' / 'Raw_Counts_Samples'
    default_suffs = ['_nr_1_a', '_nr_1_b', '_nr_2_a', '_nr_2_b']
    special = {'P38': ['_nr_1_a', '_nr_2_a', '_nr_2_b']}

    # concatenate raw replicates per age
    raw_age_adatas = []
    for age in age_ids:
        suffs = special.get(age, default_suffs)
        reps = [
            sc.read_10x_mtx(
                base / f"{age}{s}" / "filtered_feature_bc_matrix",
                var_names='gene_symbols',
                cache=True
            )
            for s in suffs
        ]
        labels = [f"{age}_{s.split('_nr_')[-1].replace('_', '')}" for s in suffs]
        raw_age_adatas.append(
            anndata.concat(reps, label='batch', keys=labels)
        )

    # full raw object
    adata_raw = raw_age_adatas[0].concatenate(*raw_age_adatas[1:], batch_categories=age_ids)

    # compute QC metrics on raw
    mito_mask = adata_raw.var_names.str.startswith('mt-')
    adata_raw.obs['percent_mito'] = (
        np.sum(adata_raw[:, mito_mask].X, axis=1).A1 /
        np.sum(adata_raw.X, axis=1).A1
    )
    adata_raw.obs['n_counts'] = adata_raw.X.sum(axis=1).A1
    adata_raw.obs['n_genes'] = (adata_raw.X > 0).sum(axis=1).A1

    # before QC plots
    sc.pl.violin(adata_raw, ['n_genes', 'n_counts', 'percent_mito'],
                 jitter=0.4, multi_panel=True)
    sc.pl.scatter(adata_raw, x='n_counts', y='percent_mito')
    sc.pl.scatter(adata_raw, x='n_counts', y='n_genes')

    # apply QC filters and plot after
    adata = apply_qc_filters(adata_raw.copy())
    sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
                 jitter=0.4, multi_panel=True)
    sc.pl.scatter(adata, x='n_counts', y='percent_mito')

    # normalize, log-transform, set raw
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    adata.write_h5ad('PooledMVC_preHVG.h5ad')
    return adata


def _compute_cluster(
    path_to_pooled_pre_HVG: str,
    hvg_params: dict,
    pca_params: dict,
    neigh_params: dict,
    leiden_params: dict,
    umap_params: dict
):
    adata = sc.read_h5ad(path_to_pooled_pre_HVG)
    sc.pp.highly_variable_genes(adata, **hvg_params)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, **pca_params)
    sc.pp.neighbors(adata, **neigh_params)
    sc.tl.leiden(
        adata,
        flavor='igraph',
        directed=False,
        n_iterations=2,
        **leiden_params
    )
    sc.tl.umap(adata, **umap_params)
    return adata

def cluster(
    path_to_pooled_pre_HVG: str,
    output_path: str = 'PooledMVC_clusteredPCA.h5ad',
    hvg_params: dict     = None,
    pca_params: dict     = None,
    neigh_params: dict   = None,
    leiden_params: dict  = None,
    umap_params: dict    = None
):
    """
    Cached Leiden‐UMAP pipeline.  Uses the hashed combination of
    path_to_pooled_pre_HVG and all param dicts to name its cache file.
    """
    hvg_params   = hvg_params   or {'min_mean':0.0125,'max_mean':3,'min_disp':0.5}
    pca_params   = pca_params   or {'svd_solver':'arpack'}
    neigh_params = neigh_params or {'n_neighbors':25,'n_pcs':40}
    leiden_params= leiden_params or {}
    umap_params  = umap_params  or {}

    key = (
        path_to_pooled_pre_HVG,
        frozenset(hvg_params.items()),
        frozenset(pca_params.items()),
        frozenset(neigh_params.items()),
        frozenset(leiden_params.items()),
        frozenset(umap_params.items())
    )
    cache_file = _cache_path('clustered', *key)
    adata = _load_or_run(cache_file,
                         lambda: _compute_cluster(
                             path_to_pooled_pre_HVG,
                             hvg_params, pca_params,
                             neigh_params, leiden_params,
                             umap_params
                         ))
    adata.write_h5ad(output_path)
    return adata

from typing import Dict, List

def post_clustering_analysis(
    input_h5ad: Path = Path('PooledMVC_preHVG.h5ad'),
    output_h5ad: Path = Path('PooledMVC_clusteredPCA.h5ad'),
    hvg_params: Dict = None,
    pca_params: Dict = None,
    neigh_params: Dict = None,
    leiden_params: Dict = None,
    umap_params: Dict = None,
    dotplot_groups: Dict[str, List[str]] = None
) -> None:
    """
    Load pre-HVG AnnData, run HVG → PCA → neighbors → Leiden → UMAP,
    save clustered object, then generate dotplots for provided marker sets.
    """
    sc.settings.verbosity = 3
    adata = sc.read_h5ad(input_h5ad)

    # default parameters
    hvg_params     = hvg_params     or {'min_mean':0.0125, 'max_mean':3, 'min_disp':0.5}
    pca_params     = pca_params     or {'svd_solver':'arpack'}
    neigh_params   = neigh_params   or {'n_neighbors':25, 'n_pcs':40}
    leiden_params  = leiden_params  or {}
    umap_params    = umap_params    or {}

    # clustering pipeline
    sc.pp.highly_variable_genes(adata, **hvg_params)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, **pca_params)
    sc.pp.neighbors(adata, **neigh_params)
    sc.tl.leiden(
        adata,
        flavor='igraph',
        directed=False,
        n_iterations=2,
        **leiden_params
    )
    sc.tl.umap(adata, **umap_params)

    adata.write_h5ad(output_h5ad)

    # dotplot marker sets
    dotplot_groups = dotplot_groups or {
        'non_neuron': [
            'Aqp4','Aldoc','Aldh1l1','S100b','Pdgfra','Enpp6',
            'Dcn','Bgn','Aox3','Osr1','Lum','Col1a1',
            'Kcnj8','Abcc9','Art3','Acta2','Cspg4','Angpt1',
            'Des','Pdgfb','Nos3','Pecam1','Tek','Mrc1',
            'Lyve1','Cd163','Cd4','Lyz2','Tmem119','Ctss','Cx3cr1'
        ],
        'gluta': [
            'Slc17a7','Snap25','Cux2','Rorb','Mdga1','Deptor',
            'Rbp4','Batf3','Osr1','Cdh9','Bcl6','Fam84b','Foxp2',
            'Slc17a8','Trhr','Sla2','Rapgef3','Ctgf','Nxph4','Inpp4b'
        ],
        'gaba': [
            'Gad1','Gad2','Sst','Cpne5','Tac1','Pvalb','Vip',
            'Pax6','Tmem182','Plch2','Dock5','Krt73','Lamp5'
        ]
    }

    for name, genes in dotplot_groups.items():
        present_genes = [g for g in genes if g in adata.var_names]
        missing_genes = [g for g in genes if g not in adata.var_names]
        print(f"Genes not present: {missing_genes}")
        sc.pl.dotplot(
            adata, present_genes,
            groupby='leiden',
            dendrogram=True,
            save=f'_{name}_pca.pdf'
        )


def process_replicate(
    matrix_path: Path,
    min_genes: int = 700,
    min_cells: int = 8,
    percent_mito_thresh: float = 0.01,
    max_genes: int = 6500,
    max_counts: int = 40000,
    expected_doublet_rate: float = 0.06
) -> AnnData:
    """
    Load a 10x matrix, filter cells/genes, compute QC metrics,
    run Scrublet and annotate doublets.
    """
    ad = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols', cache=True)
    ad = apply_qc_filters(ad, min_genes, min_cells, percent_mito_thresh, max_genes, max_counts)

    scrub = scrublet.Scrublet(ad.X, expected_doublet_rate=expected_doublet_rate)
    scrub.scrub_doublets(min_cells=min_cells, min_gene_variability_pctl=85, n_prin_comps=40)
    ad.obs['Doublet'] = scrub.predicted_doublets_
    ad.obs['Doublet Score'] = scrub.doublet_scores_obs_

    logger.debug("Processed %s, %d cells remain", matrix_path.name, ad.n_obs)
    return ad

def doublet_detection(timepoints: List[str] = ['P8', 'P14', 'P17', 'P21', 'P28', 'P38']):
    """
    Iterate over timepoints and replicates, merge per-age and then all ages.
    """
    base_path = Path(__file__).parent.parent / 'ProcessedData'
    default_suffs: List[str] = ['_nr_1_a', '_nr_1_b', '_nr_2_a', '_nr_2_b']
    special_suffs: Dict[str, List[str]] = {'P38': ['_nr_1_a', '_nr_2_a', '_nr_2_b']}

    age_adatas: List[AnnData] = []
    for tp in timepoints:
        suffixes = special_suffs.get(tp, default_suffs)
        reps = [
            process_replicate(base_path / "Raw_Counts_samples"/f"{tp}{s}" / "filtered_feature_bc_matrix")
            for s in suffixes
        ]
        batch_labels = [f"{tp}_{s.split('_nr_')[1].replace('_','')}" for s in suffixes]
        age_adatas.append(
            anndata.concat(reps, label='batch', keys=batch_labels)
        )
    adata = anndata.concat(age_adatas, label='batch', keys=timepoints)
    adata.write_h5ad('PooledMVC_dubs.h5ad')
    return

def annotate_doublets(
    clustered_path: Path,
    dubs_path: Path,
    output_path: Path = Path('PooledMVC_clusteredPCA_dubs.h5ad')
) -> None:
    """
    Load a clustered AnnData from `clustered_path`, pull in doublet
    labels/scores from `dubs_path`, create UMAP plots, and write out
    the merged object to `output_path`.
    """
    # load objects
    adata = sc.read_h5ad(clustered_path)
    adata.obs_names_make_unique()            # ensure unique barcodes

    dubs = sc.read_h5ad(dubs_path)
    dubs.obs_names_make_unique()             # ensure unique barcodes

    # intersect cell barcodes and subset
    common = adata.obs_names.intersection(dubs.obs_names)
    adata = adata[common].copy()

    # add doublet annotations
    adata.obs['Doublet'] = dubs.obs.loc[common, 'Doublet']
    adata.obs['Doublet Score'] = dubs.obs.loc[common, 'Doublet Score']

    # generate UMAPs
    sc.pl.umap(
        adata,
        color=['Doublet', 'leiden', 'Doublet Score'],
        legend_loc='on data',
        legend_fontsize='6',
        save='_DoubletView.pdf'
    )

    # write merged AnnData
    adata.write_h5ad(output_path)

def plot_doublet_score_distribution(
    input_h5ad: Path,
    output_score_pdf: Path = Path('figures/DoubletScores.pdf'),
    output_percent_pdf: Path = Path('figures/DoubletPercents.pdf')
) -> None:
    """
    Load clustered AnnData, compute per-cluster doublet score means ± SD
    and percent doublets, then plot and save bar charts.
    """
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(input_h5ad)
    # make cluster column from leiden so dendrogram can use it
    adata.obs['cluster'] = adata.obs['leiden']

    sc.tl.dendrogram(adata, groupby='cluster', key_added='dendrogram_cluster')
    clusters = adata.uns["dendrogram_cluster"]['dendrogram_info']['ivl']

    means, sds, perc = [], [], []
    for cl in clusters:
        sub = adata[adata.obs.cluster == cl]
        ds = sub.obs['Doublet Score']
        means.append(ds.mean())
        sds.append(ds.std())
        perc.append(sub.obs['Doublet'].sum() * 100 / sub.n_obs)

    # Doublet Score plot
    fig, ax = plt.subplots(figsize=(24, 9))
    ax.bar(clusters, means, yerr=sds, align='center',
           alpha=0.5, ecolor='black', capsize=5, color='blue')
    ax.set_xticklabels(clusters, rotation=45)
    ax.set_ylabel('Doublet Score')
    plt.tight_layout()
    plt.savefig(output_score_pdf)

    # Doublet Percentage plot
    fig, ax = plt.subplots(figsize=(24, 9))
    ax.bar(clusters, perc, color='blue', width=0.75)
    ax.set_xticklabels(clusters, rotation=90)
    ax.set_ylabel('Doublet Percentage')
    plt.tight_layout()
    plt.savefig(output_percent_pdf)


def annotate_clusters_and_plot(
        input_path: Path,
        output_path: Path,
        cluster_map: Dict[int, str],  # This is cluster_dict from definitions.py {int_leiden_id: named_cluster}
        class_broad_map: Dict[str, str],  # This MUST BE {str_leiden_id: broad_class_name} from updated definitions.py
        leiden_markers: List[str]
) -> None:
    """
    Load clustered AnnData, rename Leiden clusters to named clusters (in 'cluster' column),
    assign broad cell classes (in 'Class_broad' column) based on Leiden IDs,
    write out, then plot UMAPs (Class_broad, leiden, cluster, batch, Doublet) and a dotplot.
    """
    from matplotlib.colors import ListedColormap
    import seaborn as sns

    adata = sc.read_h5ad(input_path)
    adata = adata[adata.obs['leiden'] != '42', :]  # Filter out cluster '42'

    # 1. Rename Leiden clusters to named clusters for adata.obs['cluster']
    adata.obs['cluster'] = adata.obs['leiden']  # Create 'cluster' from 'leiden'
    # Prepare map for renaming: cluster_map keys (int) to string to match leiden categories (str)
    new_category_names_list = [cluster_map[int(cat)] for cat in adata.obs['leiden'].cat.categories]
    adata.rename_categories(key='cluster', categories=new_category_names_list)

    # Now adata.obs['cluster'] contains names like 'Exc_13', 'Astro_2'

    # 2. Assign broad cell classes to adata.obs['Class_broad']
    # Uses the original 'leiden' column (e.g., '0', '6') and the
    # class_broad_map (which MUST be {str_leiden_id: broad_class_name} in definitions.py).
    mapped_broad_classes = adata.obs['leiden'].astype(str).map(class_broad_map)
    adata.obs['Class_broad'] = mapped_broad_classes.fillna('Excitatory').astype('category')
    # Now adata.obs['Class_broad'] contains 'Astrocytes', 'Excitatory', etc.

    # 3. (Recommended) Set category order for 'Class_broad' for consistent plotting
    broad_class_order = [
        'Astrocytes', 'Oligodendrocytes', 'OPCs', 'Microglia',
        'Endothelial', 'VLMC', 'Inhibitory', 'Excitatory', 'Ambiguous'
    ]
    # Filter to only include classes actually present in the data
    present_broad_classes = [c for c in broad_class_order if c in adata.obs['Class_broad'].cat.categories]
    if present_broad_classes:  # Check if list is not empty
        adata.obs['Class_broad'] = adata.obs['Class_broad'].cat.set_categories(present_broad_classes)

    # 4. Save annotated AnnData
    adata.write_h5ad(output_path)

    # 5. UMAP Palettes (improved for dynamic number of categories)
    num_actual_broad_classes = len(adata.obs['Class_broad'].cat.categories)
    palette_cb_base = sns.color_palette('Set1', n_colors=max(1, num_actual_broad_classes))
    if num_actual_broad_classes > len(palette_cb_base):
        palette_cb = (sns.color_palette('Set1').as_hex() +
                      sns.color_palette('Set2').as_hex() +
                      sns.color_palette('Set3').as_hex())[:num_actual_broad_classes]
    else:
        palette_cb = palette_cb_base.as_hex()[:num_actual_broad_classes]

    # Palette for Leiden clusters
    num_leiden_cats = len(adata.obs['leiden'].cat.categories)
    base_palettes_long = (sns.color_palette('pastel').as_hex() +
                          sns.color_palette('Set2').as_hex() +
                          sns.color_palette('Set3').as_hex() +
                          sns.color_palette('hls').as_hex() +
                          sns.color_palette('husl').as_hex())
    leiden_cluster_cols = base_palettes_long[:num_leiden_cats]
    if num_leiden_cats > len(leiden_cluster_cols):  # Repeat if not enough unique colors
        leiden_cluster_cols = (base_palettes_long * (num_leiden_cats // len(base_palettes_long) + 1))[:num_leiden_cats]

    # Palette for named clusters (adata.obs['cluster'])
    num_named_clusters = len(adata.obs['cluster'].cat.categories)
    named_cluster_cols = base_palettes_long[:num_named_clusters]
    if num_named_clusters > len(named_cluster_cols):
        named_cluster_cols = (base_palettes_long * (num_named_clusters // len(base_palettes_long) + 1))[
                             :num_named_clusters]

    # 6. UMAP plots
    sc.pl.umap(adata, color=['Class_broad'], palette=palette_cb,
               add_outline=True, save='_Classes_broad.pdf')  # This should now be correct

    sc.pl.umap(adata, color='leiden', palette=leiden_cluster_cols,  # Original Leiden IDs
               legend_loc='on data', legend_fontsize='6',
               add_outline=True, save='_leiden.pdf')

    sc.pl.umap(adata, color='cluster', palette=named_cluster_cols,  # Named clusters
               legend_loc='on data', legend_fontsize='6',
               add_outline=True, save='_cluster_named.pdf')

    sc.pl.umap(adata, color='batch', legend_fontsize='10',  # 'batch' likely refers to age
               add_outline=True, save='_age.pdf')

    sc.pl.umap(adata, color='Doublet',
               legend_loc='on data', legend_fontsize='6',
               add_outline=True, save='_DoubletsColor.pdf')

    # 7. Dotplot for main markers
    # groupby='cluster' uses the named clusters (e.g., 'Exc_13', 'Astro_1'), which is good.
    mapcol = ListedColormap(sns.color_palette('light:#a31fe7', n_colors=100).as_hex())
    sc.pl.dotplot(adata, leiden_markers, groupby='cluster',
                  dendrogram=True, dot_max=0.8, vmax=2,
                  cmap=mapcol, save='Markers_cluster.pdf')


def assign_subclasses(
    input_h5ad: Path,
    subclass_map: Dict[str, List[str]],
    output_h5ad: Path
) -> None:
    """
    Load AnnData, map clusters to subclasses based on subclass_map,
    then write updated AnnData to output_h5ad.
    """
    import pandas as pd

    adata = sc.read_h5ad(input_h5ad)
    # invert subclass_map for fast lookup
    cluster_to_sub = {
        cluster: sub
        for sub, clusters in subclass_map.items()
        for cluster in clusters
    }
    adata.obs['Subclass'] = pd.Categorical(
        adata.obs['cluster'].map(lambda cl: cluster_to_sub.get(cl, 'Unknown'))
    )
    adata.write_h5ad(output_h5ad)


def plot_subclass_doublet_distribution(
    input_h5ad: Path,
    output_score_pdf: Path = Path('figures/Subclass_DoubletScores.pdf'),
    output_percent_pdf: Path = Path('figures/Subclass_DoubletPercents.pdf')
) -> None:
    """
    Load annotated AnnData with Subclass, compute per-subclass
    doublet score means ± SD and percent doublets, then plot/bar charts.
    """
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(input_h5ad)
    sc.tl.dendrogram(adata, groupby='Subclass', key_added='dendrogram_Subclass')
    subclasses = adata.uns["dendrogram_Subclass"]['dendrogram_info']['ivl']

    means, sds, perc = [], [], []
    for sub in subclasses:
        subdata = adata[adata.obs['Subclass'] == sub]
        ds = subdata.obs['Doublet Score']
        means.append(ds.mean())
        sds.append(ds.std())
        perc.append(subdata.obs['Doublet'].sum() * 100 / subdata.n_obs)

    # Doublet Score plot
    fig, ax = plt.subplots(figsize=(24, 9))
    ax.bar(subclasses, means, yerr=sds, align='center',
           alpha=0.5, ecolor='black', capsize=5, color='blue', width=0.75)
    ax.set_ylabel('Doublet Score')
    ax.set_xticklabels(subclasses, rotation=45)
    plt.tight_layout()
    plt.savefig(output_score_pdf)

    # Doublet Percentage plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(subclasses, perc, color='blue', width=0.75)
    ax.set_ylabel('Doublet Percentage')
    ax.set_xticklabels(subclasses)
    plt.tight_layout()
    plt.savefig(output_percent_pdf)


def plot_cluster_size_distribution(
        input_h5ad: Path,
        output_pdf: Path,
        figsize: tuple
) -> None:
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(input_h5ad)
    counts = adata.obs.cluster.value_counts(normalize=True) * 100
    order = list(adata.obs.cluster.cat.categories)

    # fallback to leiden_colors if cluster_colors is missing
    colors = adata.uns.get('cluster_colors', adata.uns.get('leiden_colors'))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(order, counts.reindex(order, fill_value=0),
           color=colors, edgecolor='black', linewidth=1)
    ax.spines[['right','top','bottom']].set_visible(False)
    ax.set_ylabel('Percentage of cells')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.savefig(output_pdf)
    plt.close()

def remove_doublets_and_ambiguous(
        input_h5ad: Path,
        output_h5ad: Path = Path('PooledMVC_clusteredPCA_dubs_classes_clean.h5ad'),
        ambiguous_prefixes: List[str] = ['Ambig_'],
        verbose: bool = True
) -> AnnData:
    """
    Remove ambiguous clusters and doublet cells to create a cleaned dataset.

    Args:
        input_h5ad: Path to annotated AnnData with clusters and doublet annotations
        output_h5ad: Path to write clean AnnData
        ambiguous_prefixes: List of prefixes identifying ambiguous clusters
        verbose: Whether to print object size before/after cleaning

    Returns:
        Cleaned AnnData object
    """
    adata = sc.read_h5ad(input_h5ad)

    if verbose:
        print('Object before removing doublets clusters')
        print(adata)

    # Filter out ambiguous clusters
    mask = ~adata.obs['cluster'].str.startswith(tuple(ambiguous_prefixes))
    adata = adata[mask]

    # Remove doublet cells
    adata = adata[adata.obs['Doublet'] == False]

    if verbose:
        print('Object after removing doublets clusters')
        print(adata)
        print(f"Number of cells remaining: {adata.shape[0]}")

    adata.write_h5ad(output_h5ad)
    return adata


def plot_age_class_distributions(
        input_h5ad: Path,
        output_class_pdf: Path = Path('figures/age_fraction_class.pdf'),
        output_age_pdf: Path = Path('figures/age_fraction_cluster.pdf'),
        figsize_class: tuple = (20, 7),
        figsize_age: tuple = (30, 8)
) -> None:
    """
    Create stacked bar charts showing:
    1. The distribution of broad cell classes within each age group (batch).
    2. The distribution of age groups (batch) within each named cluster.

    Pulls colors from adata.uns['Class_broad_colors'] and adata.uns['batch_colors'].
    Orders clusters based on adata.uns["dendrogram_['cluster']"].
    """
    adata = sc.read_h5ad(input_h5ad)
    obs = adata.obs

    # --- Plot 1: Class distribution across Ages ---
    # Define the desired order for broad classes (as in paste-2.txt)
    # This order will also dictate the legend and stacking order.
    class_order = ['Astrocytes', 'Oligodendrocytes', 'OPCs', 'Microglia',
                   'Endothelial', 'VLMC', 'Inhibitory', 'Excitatory', 'Ambiguous']

    # Ensure 'Class_broad' and 'batch' are categorical
    if not pd.api.types.is_categorical_dtype(obs['Class_broad']):
        obs['Class_broad'] = obs['Class_broad'].astype('category')
    if not pd.api.types.is_categorical_dtype(obs['batch']):
        obs['batch'] = obs['batch'].astype('category')

    # Calculate proportion of each Class_broad for each batch (age)
    ct1 = pd.crosstab(obs.batch, obs.Class_broad, normalize='index') * 100

    # Reorder columns (Class_broad) to match class_order and fill missing with 0
    # Also, only include classes present in the data to avoid issues with missing colors/categories
    plot_class_order = [c for c in class_order if c in ct1.columns]
    ct1 = ct1.reindex(columns=plot_class_order, fill_value=0)

    fig, ax = plt.subplots(figsize=figsize_class)

    # Prepare colors for Class_broad, ensuring alignment with plot_class_order
    class_broad_plot_colors = []
    if 'Class_broad_colors' in adata.uns and adata.obs['Class_broad'].cat.categories.any():
        # Create a mapping from actual category name to its color
        class_cat_list = adata.obs['Class_broad'].cat.categories.tolist()
        color_map_dict = dict(zip(class_cat_list, adata.uns['Class_broad_colors']))
        class_broad_plot_colors = [color_map_dict.get(cls, '#CCCCCC') for cls in
                                   plot_class_order]  # Use gray for missing
    else:
        logger.warning(
            "adata.uns['Class_broad_colors'] not found or 'Class_broad' has no categories. Using fallback palette.")
        if len(plot_class_order) > 0:
            class_broad_plot_colors = sns.color_palette('tab20', n_colors=len(plot_class_order)).as_hex()

    # Plot using pandas' plot method with specified colors
    if not ct1.empty and class_broad_plot_colors:
        ct1.plot(kind='bar', stacked=True, ax=ax, width=0.5,
                 edgecolor='white', color=class_broad_plot_colors)
        ax.legend(title='Broad Class', ncol=max(1, len(plot_class_order) // 2), bbox_to_anchor=(0, 1.02),
                  loc='lower left', fontsize='small')
    else:
        ax.text(0.5, 0.5, "No data to plot for class distribution by age.", ha='center', va='center')

    ax.set_ylabel("Percentage of Cells per Age Group")
    ax.set_xlabel("Age Group (Batch)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_class_pdf)
    plt.close()

    # --- Plot 2: Age distribution across Clusters ---
    # Get cluster order from dendrogram
    if "dendrogram_['cluster']" in adata.uns and 'ivl' in adata.uns["dendrogram_['cluster']"]['dendrogram_info']:
        cluster_order = list(adata.uns["dendrogram_['cluster']"]['dendrogram_info']['ivl'])
    else:
        logger.warning("Dendrogram for 'cluster' not found or 'ivl' key missing. Using sorted cluster names.")
        if pd.api.types.is_categorical_dtype(obs['cluster']):
            cluster_order = sorted(list(obs.cluster.cat.categories))
        else:  # Fallback if cluster is not categorical or missing
            cluster_order = sorted(list(obs.cluster.unique()))

    # Ensure 'cluster' is categorical for crosstab
    if not pd.api.types.is_categorical_dtype(obs['cluster']):
        obs['cluster'] = obs['cluster'].astype('category')

    # Calculate proportion of each batch (age) for each cluster
    ct2 = pd.crosstab(obs.cluster, obs.batch, normalize='index') * 100

    # Define desired age order for columns (as in paste-2.txt)
    age_order_plot = ['P8', 'P14', 'P17', 'P21', 'P28', 'P38']
    # Filter age_order_plot to only include ages present in ct2.columns
    actual_age_order_for_plot = [age for age in age_order_plot if age in ct2.columns]

    # Reorder columns (batch/age) and rows (cluster)
    ct2 = ct2.reindex(columns=actual_age_order_for_plot, fill_value=0)
    ct2 = ct2.reindex(index=cluster_order, fill_value=0)  # Match cluster order from dendrogram

    fig, ax = plt.subplots(figsize=figsize_age)

    # Prepare colors for batch/age, ensuring alignment with actual_age_order_for_plot
    batch_plot_colors = []
    if 'batch_colors' in adata.uns and adata.obs['batch'].cat.categories.any():
        batch_cat_list = adata.obs['batch'].cat.categories.tolist()
        color_map_dict_batch = dict(zip(batch_cat_list, adata.uns['batch_colors']))
        batch_plot_colors = [color_map_dict_batch.get(age, '#CCCCCC') for age in actual_age_order_for_plot]
    else:
        logger.warning("adata.uns['batch_colors'] not found or 'batch' has no categories. Using fallback palette.")
        if len(actual_age_order_for_plot) > 0:
            batch_plot_colors = sns.color_palette('viridis', n_colors=len(actual_age_order_for_plot)).as_hex()

    if not ct2.empty and batch_plot_colors and len(batch_plot_colors) == len(ct2.columns):
        bottom = pd.Series(0.0, index=ct2.index)  # Initialize bottom for stacking
        for i, age_column_name in enumerate(ct2.columns):  # Iterate through ages in the order of ct2.columns
            vals = ct2[age_column_name]
            ax.bar(ct2.index, vals, bottom=bottom, color=batch_plot_colors[i],
                   edgecolor='white', width=0.5, label=age_column_name)
            bottom += vals
        ax.legend(title='Age Group', loc='best', fontsize='x-large')
    else:
        ax.text(0.5, 0.5, "No data to plot for age distribution by cluster.", ha='center', va='center')
        if not batch_plot_colors or len(batch_plot_colors) != len(ct2.columns):
            logger.error(f"Color mismatch: {len(batch_plot_colors)} colors for {len(ct2.columns)} age categories.")

    ax.set_xticklabels(ct2.index, rotation=90, ha='right')  # Use ct2.index for tick labels
    ax.set_ylabel("Percentage of Cells per Cluster")
    ax.set_xlabel("Cluster")
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Y-axis grid is often more useful for proportions
    plt.tight_layout()
    plt.savefig(output_age_pdf)
    plt.close()


def plot_sample_class_distribution(
        input_h5ad: Path = Path('gluta_gaba_glia_combined_analyzed.h5ad'),
        output_csv: Path = Path('data/samples_barplot.csv'),
        output_pdf: Path = Path('figures/sample_dist_Fig1.pdf'),
        figsize: tuple = (14, 6),
        class_categories: List[str] = ['Excitatory', 'Inhibitory', 'Non-neuron']
) -> None:
    """
    Create a stacked bar chart showing cell type proportions across samples.

    Args:
        input_h5ad: Path to analyzed h5ad file containing 'sample' and 'Class' annotations
        output_csv: Path to save the proportion data as CSV
        output_pdf: Path to save the stacked bar chart
        figsize: Figure dimensions in inches
        class_categories: List of cell class names to plot (in stacking order)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(input_h5ad)

    # Calculate proportions by sample
    proportions = {}
    for sample_name in adata.obs['sample'].values.categories:
        sample_subset = adata[adata.obs['sample'] == sample_name]
        counts = sample_subset.obs.Class.value_counts(normalize=True)
        proportions[sample_name] = [
            counts.get(class_name, 0) for class_name in class_categories
        ]

    # Create and save DataFrame
    proportion_df = pd.DataFrame(
        proportions,
        index=class_categories
    ).transpose()

    proportion_df.to_csv(output_csv)

    # Plot stacked bars
    proportion_df.plot(
        kind='bar',
        stacked=True,
        mark_right=True,
        grid=False,
        linewidth=0.5,
        width=0.9,
        figsize=figsize
    )

    plt.legend(bbox_to_anchor=(0.99, 1), loc='upper left', fontsize=14, ncol=1)
    plt.ylabel('Proportion of Sample', fontsize=16)
    plt.xlabel('Sample', rotation='180', fontsize=16)
    plt.yticks([0, 0.25, 0.50, 0.75, 1.00], rotation='90')
    plt.xticks(rotation='90')
    plt.tight_layout()
    plt.savefig(output_pdf, dpi=200)
    plt.close()


def plot_age_distribution_by_cluster_reversed(
        input_h5ad: Path,
        output_pdf: Path = Path('figures/age_fraction_reversed.pdf'),
        figsize: tuple = (30, 8),
        barwidth: float = 0.5
) -> None:
    """
    Create a stacked bar chart showing age distribution across clusters
    with P8 at the top of the stack and P38 at the bottom.

    Args:
        input_h5ad: Path to annotated AnnData with clusters and age information
        output_pdf: Path to save the age distribution chart
        figsize: Figure dimensions for the chart
        barwidth: Width of the bars in the chart
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    adata = sc.read_h5ad(input_h5ad)

    # Ensure 'cluster' is used as groupby for dendrogram
    sc.tl.dendrogram(adata, groupby='cluster', key_added='dendrogram_cluster')

    # Get clusters in dendrogram order
    clusters = list(adata.uns['dendrogram_cluster']['dendrogram_info']['ivl'])

    # Get actual ages present in the data
    available_ages = adata.obs.batch.cat.categories.tolist()

    # Use standard age order but only include ages present in the data
    expected_ages = ['P8', 'P14', 'P17', 'P21', 'P28', 'P38']
    ages = [age for age in expected_ages if age in available_ages]

    if not ages:  # Fallback if no expected ages match
        ages = available_ages

    # Create a cross-tabulation to count cells per cluster and age
    cross_tab = pd.crosstab(
        adata.obs['cluster'],
        adata.obs['batch']
    )

    # Ensure we only use ages that exist in the data
    cross_tab = cross_tab.reindex(columns=ages, fill_value=0)

    # Reorder rows according to dendrogram
    cross_tab = cross_tab.reindex(index=clusters)

    # Calculate proportion per cluster (normalize rows)
    proportion_tab = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    # Get colors from AnnData or use a fallback palette
    if 'batch_colors' in adata.uns and len(adata.uns['batch_colors']) >= len(available_ages):
        color_map = dict(zip(available_ages, adata.uns['batch_colors']))
        colors = [color_map.get(age, '#CCCCCC') for age in ages]
    else:
        # Fallback to viridis palette
        colors = sns.color_palette('viridis', n_colors=len(ages))

    # Create stacked bar chart
    plt.figure(figsize=figsize)
    r = range(len(clusters))

    # Define stacking order and plot from bottom (P38) to top (P8)
    reverse_ages = list(reversed(ages))

    bottom = np.zeros(len(clusters))
    for i, age in enumerate(reverse_ages):
        age_idx = ages.index(age)  # Get original index for color mapping

        plt.bar(r, proportion_tab[age],
                bottom=bottom,
                color=colors[age_idx],
                edgecolor='white',
                width=barwidth,
                label=age)

        # Update bottom for next bar
        bottom += proportion_tab[age]

    plt.legend(loc='best', fontsize='x-large')
    plt.xticks(r, clusters, rotation=90)
    plt.ylabel("Percentage of Cells per Cluster")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

def split_by_class_and_age(
        input_h5ad: Path = Path('PooledMVC_clusteredPCA_dubs_classes_clean.h5ad'),
        output_dir: Path = Path('data/split_objects'),
        neuron_classes: List[str] = ['Excitatory', 'Inhibitory'],
        verbose: bool = True
) -> Dict[str, Path]:
    """
    Split AnnData by cell class and age timepoints, saving class-specific objects
    and age-specific objects for neuronal classes.
    """
    import os

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    adata = sc.read_h5ad(input_h5ad)

    if verbose:
        print(f"Loaded dataset with {adata.n_obs} cells")

    # Define the output paths
    class_paths = {}
    age_paths = {}

    # Split by major classes
    class_objects = {}

    # Create neuron class objects
    for class_name in neuron_classes:
        class_objects[class_name] = adata[adata.obs['Class_broad'] == class_name].copy()
        class_paths[class_name] = output_dir / f"{class_name.lower()}.h5ad"

    # Create non-neuronal object (cells that aren't in any neuron class)
    non_mask = ~adata.obs['Class_broad'].isin(neuron_classes)
    class_objects['Non-neuron'] = adata[non_mask].copy()
    class_paths['Non-neuron'] = output_dir / "non_neuron.h5ad"

    # Verify split was successful
    total_cells = sum(obj.n_obs for obj in class_objects.values())
    if total_cells == adata.n_obs and verbose:
        print(f"Split successful: {total_cells} cells distributed across classes")
    elif verbose:
        print(f"Warning: Split resulted in {total_cells} cells vs {adata.n_obs} in original")

    # Save class objects
    for class_name, obj in class_objects.items():
        if verbose:
            print(f"Saving {class_name} object with {obj.n_obs} cells")
        obj.write_h5ad(class_paths[class_name])

    # Create age-specific objects for neuronal classes
    # FIXED: Correctly access categorical categories
    ages = list(adata.obs.batch.cat.categories)

    for class_name in neuron_classes:
        class_obj = class_objects[class_name]

        for age in ages:
            # Extract cells for this class and age
            age_obj = class_obj[class_obj.obs.batch == age].copy()

            # Simplify object by removing unnecessary slots
            for slot in ('obsp', 'varm', 'obsm', 'uns'):
                if hasattr(age_obj, slot):
                    setattr(age_obj, slot, {})

            # Use raw counts
            if hasattr(age_obj, 'raw') and age_obj.raw is not None:
                age_obj.X = age_obj.raw.X.copy()

            # Save age-specific object
            age_path = output_dir / f"{age}_{class_name.lower()}_raw.h5ad"
            age_obj.write_h5ad(age_path)
            age_paths[f"{age}_{class_name}"] = age_path

            if verbose:
                print(f"  Saved {age} {class_name} object with {age_obj.n_obs} cells")

    # Return all paths
    return {**class_paths, **age_paths}


if __name__ == '__main__':
    from pathlib import Path

    # 1. Import and filter 10x for one timepoint
    datasets = ['P28']
    print(f"Analyzing: {datasets}")
    adata_pre = import_10x(datasets)

    # 2. Cluster and save to `PooledMVC_clusteredPCA.h5ad`
    cluster(
        path_to_pooled_pre_HVG='PooledMVC_preHVG.h5ad',
        output_path='PooledMVC_clusteredPCA.h5ad'
    )

    # 3. Post-clustering analysis and dotplots
    post_clustering_analysis(
        input_h5ad='PooledMVC_preHVG.h5ad',
        output_h5ad='PooledMVC_clusteredPCA.h5ad'
    )

    # 4. Doublet detection and annotation
    doublet_detection(datasets)
    annotate_doublets(
        clustered_path=Path('PooledMVC_clusteredPCA.h5ad'),
        dubs_path=Path('PooledMVC_dubs.h5ad'),
        output_path=Path('PooledMVC_clusteredPCA_dubs.h5ad')
    )

    # 5. Plot doublet score distribution on the merged AnnData
    plot_doublet_score_distribution(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs.h5ad')
    )

    # 6. Rename clusters, assign broad classes, UMAPs & dotplot
    annotate_clusters_and_plot(
        input_path=Path('PooledMVC_clusteredPCA_dubs.h5ad'),
        output_path=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad'),
        cluster_map=cluster_dict,
        class_broad_map=class_broad_map,
        leiden_markers=leiden_markers
    )

    # 7. Assign subclasses and plot their doublet distributions
    assign_subclasses(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad'),
        subclass_map=subclass_map,
        output_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad')
    )
    plot_subclass_doublet_distribution(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad')
    )

    remove_doublets_and_ambiguous(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad'),
        output_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes_clean.h5ad')
    )

    plot_cluster_size_distribution(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad'),
        output_pdf=Path('figures/cluster_fraction.pdf'),
        figsize=(20, 7)
    )
    plot_age_class_distributions(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad'),
        output_class_pdf=Path('figures/age_fraction_class.pdf'),
        output_age_pdf=Path('figures/age_fraction_cluster.pdf')
    )
    if 0:
        plot_sample_class_distribution(
            input_h5ad=Path('gluta_gaba_glia_combined_analyzed.h5ad'),
            output_csv=Path('data/samples_barplot.csv'),
            output_pdf=Path('figures/sample_dist_Fig1.pdf')
        )
    plot_age_distribution_by_cluster_reversed(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes.h5ad'),
        output_pdf=Path('figures/age_fraction_by_cluster.pdf')
    )
    result_paths = split_by_class_and_age(
        input_h5ad=Path('PooledMVC_clusteredPCA_dubs_classes_clean.h5ad'),
        output_dir=Path('data/split_objects')
    )