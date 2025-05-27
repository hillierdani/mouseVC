# map each leiden cluster index to its human-readable name
cluster_dict = {
    0: 'Exc_13', 1: 'Exc_5', 2: 'Exc_18', 3: 'Exc_4', 4: 'Exc_17',
    5: 'Exc_2', 6: 'Astro_2', 7: 'OPC_1', 8: 'Exc_19', 9: 'Inh_2',
    10: 'Micro', 11: 'OD_1', 12: 'Exc_15', 13: 'Exc_14', 14: 'Inh_5',
    15: 'Astro_3', 16: 'Ambig_7', 17: 'Exc_8', 18: 'Inh_4', 19: 'Exc_12',
    20: 'Exc_16', 21: 'Exc_9', 22: 'OD_2', 23: 'Endo', 24: 'Exc_6',
    25: 'Inh_1', 26: 'Exc_1', 27: 'Ambig_2', 28: 'Exc_11', 29: 'Ambig_6',
    30: 'Astro_1', 31: 'Ambig_4', 32: 'Exc_10', 33: 'Ambig_5', 34: 'VLMC_1',
    35: 'Exc_7', 36: 'Ambig_3', 37: 'VLMC_2', 38: 'Inh_3', 39: 'Exc_3',
    40: 'OPC_2', 41: 'Ambig_1'
}

# map each named cluster to a broad cell class
# Map Leiden string IDs directly to Broad Cell Classes.
# This dictionary is used by ClassSeparation.py to assign adata.obs['Class_broad'].
# Leiden IDs not present in this map will be defaulted to 'Excitatory'
# in ClassSeparation.py, as per the logic in paste-2.txt.
class_broad_map = {
    # Leiden ID (string) : Broad Class (string)
    '30': 'Astrocytes', '6': 'Astrocytes', '15': 'Astrocytes',
    '11': 'Oligodendrocytes', '22': 'Oligodendrocytes',
    '7': 'OPCs', '40': 'OPCs',
    '10': 'Microglia',        # Note: paste-2.txt had (10, 4000). 4000 seems to be a typo as cluster_dict only goes up to 41.
    '23': 'Endothelial',      # Note: paste-2.txt had (23, 4000). 4000 seems to be a typo.
    '34': 'VLMC', '37': 'VLMC',
    '25': 'Inhibitory', '9': 'Inhibitory', '38': 'Inhibitory', '18': 'Inhibitory', '14': 'Inhibitory',
    '16': 'Ambiguous', '27': 'Ambiguous', '29': 'Ambiguous', '31': 'Ambiguous',
    '33': 'Ambiguous', '36': 'Ambiguous', '41': 'Ambiguous',
    # Excitatory clusters are not listed here; they will be assigned by default.
}

# list of marker genes for Leiden-based dotplots
leiden_markers = [
    'Cx3cr1', 'Mog', 'Pdgfra', 'Aldh1l1', 'Pecam1', 'Slc47a1',
    'Snap25', 'Gad1', 'Vip', 'Pvalb', 'Sst', 'Lamp5',
    'Slc17a7', 'Cux2', 'Rorb', 'Deptor', 'Tshz2', 'Bcl6',
    'Foxp2', 'Cdh9'
]

# map each subclass name to the list of cluster names it contains
subclass_map = {
    'L2/3': ['Exc_1', 'Exc_2', 'Exc_17', 'Exc_18', 'Exc_19'],
    'L4':   ['Exc_3', 'Exc_12', 'Exc_13', 'Exc_14', 'Exc_16'],
    'L5':   ['Exc_8', 'Exc_9', 'Exc_15'],
    'L6':   ['Exc_4', 'Exc_5', 'Exc_6', 'Exc_7', 'Exc_10', 'Exc_11'],
    'Ambig':['Ambig_1','Ambig_2','Ambig_3','Ambig_4','Ambig_5','Ambig_6','Ambig_7'],
    'Astro':['Astro_1','Astro_2','Astro_3'],
    'OD':   ['OD_1','OD_2'],
    'OPC':  ['OPC_1','OPC_2'],
    'VLMC': ['VLMC_1','VLMC_2'],
    'Inh':  ['Inh_1','Inh_3'],
    'Micro':['Micro'],
    'Endo': ['Endo'],
    'Pvalb':['Inh_5'],
    'Sst':  ['Inh_4'],
    'Vip':  ['Inh_2']
}