#!/usr/bin/env python3
"""
ProteoForge Annotation Module


Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import re

import numpy as np
import pandas as pd


# ======================================================================================
# Global Variables
# ======================================================================================
# TODO: Add the global variables as needed

# Source: https://www.uniprot.org/help/feature_table

uniprot_feature_dict = {
    # Protein Processing & Maturation
    'INIT_MET': 'Initiator methionine',
    'SIGNAL': 'Signal peptide',
    'TRANSIT': 'Transit peptide',
    'PROPEP': 'Propeptide',
    'CHAIN': 'Chain',
    'PEPTIDE': 'Peptide',
    'CLEAVAGE': 'Proteolytic Cleavage', # Custom parsed feature

    # Co- & Post-Translational Modifications
    'MOD_RES': 'Modified residue',
    'CARBOHYD': 'Glycosylation',
    'LIPID': 'Lipidation',
    'DISULFID': 'Disulfide bond',
    'CROSSLNK': 'Cross-link',
    'NON_STD': 'Non-standard residue',

    # Sequence Heterogeneity & Isoforms
    'VARIANT': 'Natural variant',
    'VAR_SEQ': 'Alternative sequence',
    'MUTAGEN': 'Mutagenesis',
    'CONFLICT': 'Sequence conflict',
    'BREAKPOINT': 'Genetic Breakpoint', # Custom parsed feature

    # Functional Domains, Regions & Sites
    'DOMAIN': 'Domain',
    'REPEAT': 'Repeat',
    'ZN_FING': 'Zinc finger',
    'MOTIF': 'Motif',
    'REGION': 'Region',
    'ACT_SITE': 'Active site',
    'BINDING': 'Binding site',
    'SITE': 'Site',
    'DNA_BIND': 'DNA binding',
    'CA_BIND': 'Calcium binding',
    'METAL': 'Metal binding',
    'NP_BIND': 'Nucleotide binding',
    
    # Structure & Topology
    'HELIX': 'Helix',
    'STRAND': 'Beta strand',
    'TURN': 'Turn',
    'COILED': 'Coiled coil',
    'TRANSMEM': 'Transmembrane',
    'INTRAMEM': 'Intramembrane',
    'TOPO_DOM': 'Topological domain',
    'COMPBIAS': 'Compositional bias',

    # Other
    'NON_TER': 'Non-terminal residue',
    'UNSURE': 'Sequence uncertainty'
}

# Placing the UniProt features into categories
feature_categories = {
    'Protein Processing & Maturation': [
        'INIT_MET', 'SIGNAL', 'TRANSIT', 
        'PROPEP', 'CHAIN', 'PEPTIDE', 'CLEAVAGE'
    ],
    'Co- & Post-Translational Modifications': [
        'MOD_RES', 'CARBOHYD', 'LIPID', 
        'DISULFID', 'CROSSLNK', 'NON_STD'
    ],
    'Sequence Heterogeneity & Isoforms': [
        'VARIANT', 'VAR_SEQ', 'MUTAGEN', 
        'CONFLICT', 'BREAKPOINT'
    ],
    'Functional Domains, Regions & Sites': [
        'DOMAIN', 'REPEAT', 'ZN_FING', 'MOTIF', 
        'REGION', 'ACT_SITE', 'BINDING', 'SITE', 
        'DNA_BIND', 'CA_BIND', 'METAL', 'NP_BIND'
    ],
    'Structure & Topology': [
        'HELIX', 'STRAND', 'TURN', 'COILED', 
        'TRANSMEM', 'INTRAMEM', 'TOPO_DOM', 'COMPBIAS'
    ]
}

feature_hierarchy = [
    # --- 1. Protein Processing & Maturation ---
    'INIT_MET',      # N-term Met excision
    'SIGNAL',        # Signal peptide cleavage
    'TRANSIT',       # Transit peptide cleavage
    'PROPEP',        # Propeptide cleavage
    'CLEAVAGE',      # Other internal proteolytic events
    'CHAIN',         # The resulting mature chain
    'PEPTIDE',       # The resulting mature peptide
    
    # --- 2. Co- & Post-Translational Modifications ---
    'MOD_RES',       # General modified residue (PTM)
    'CARBOHYD',      # Glycosylation
    'LIPID',         # Lipidation
    'DISULFID',      # Disulfide bond 
    'CROSSLNK',      # Cross-link 
    'NON_STD',       # Non-standard residue (e.g., selenocysteine, pyrrolysine)
    
    # --- 3. Sequence Heterogeneity & Isoforms ---
    'VARIANT',       # Natural variant (polymorphism)
    'VAR_SEQ',       # Alternative sequence (e.g., isoform, splice variant)
    'MUTAGEN',       # Mutagenesis (experimental mutation)
    'CONFLICT',      # Sequence conflict (discrepancy in sequence data)
    
    # --- 4. Functional Domains, Regions & Sites ---
    'DOMAIN',       # Functional domain (e.g., kinase domain, SH3 domain)
    'REPEAT',       # Repeated sequence (e.g., leucine-rich repeat)
    'ZN_FING',      # Zinc finger motif
    'MOTIF',        # Short conserved sequence motif (e.g., PEST motif)
    'REGION',       # General region (e.g., transactivation domain)
    'ACT_SITE',     # Active site (catalytic site of an enzyme)
    'BINDING',      # Binding site (e.g., ligand binding, protein-protein interaction)
    'SITE',         # General site (e.g., phosphorylation site)
    'DNA_BIND',     # DNA binding site (e.g., transcription factor binding)
    
    # --- 5. Structure & Topology ---
    'HELIX',        # Alpha helix
    'STRAND',       # Beta strand
    'TURN',         # Beta turn
    'COILED',       # Coiled coil (e.g., leucine zipper)
    'TRANSMEM',     # Transmembrane region (e.g., integral membrane protein)
    'INTRAMEM',     # Intramembrane region (e.g., membrane-spanning helix)
    'TOPO_DOM',     # Topological domain (e.g., extracellular, cytoplasmic)
    'COMPBIAS'      # Compositional bias (e.g., low complexity region, polyglutamine tract)
]

# Ordered list of the category names (mostly for visualization purposes)
category_hierarchy = [
    'Protein Processing & Maturation',
    'Co- & Post-Translational Modifications',
    'Sequence Heterogeneity & Isoforms',
    'Functional Domains, Regions & Sites',
    'Structure & Topology'
]

# List to easily access the which features are most 
#   important/relevant for proteoform analysis
relevant_features = [
    # Protein Processing & Maturation (minus CHAIN, PROPEP, PEPTIDE, TRANSIT)
    'INIT_MET', 'SIGNAL', 'CLEAVAGE',
    # Co- & Post-Translational Modifications
    'MOD_RES', 'CARBOHYD', 'LIPID', 'DISULFID', 'CROSSLNK', 'NON_STD',
    # Sequence Heterogeneity & Isoforms
    'VARIANT', 'VAR_SEQ', 'MUTAGEN', 'CONFLICT', 'BREAKPOINT'
]

# ======================================================================================
# UniProt to Table Parsing Functions
# ======================================================================================
# Source: Alphamap/alphamap/uniprot_integration.py - adapted and modified
# Main and helper functions used as is with minimal changes

# TODO: Optimize the main function 

def extract_note(
        string: str, splitted:bool = False
    ) -> str:
    """
        Helper function to extract information about note of the 
            protein from Uniprot using regular expression.

        Args:
            string (str): Uniprot annotation string.
            splitted (bool, optional): Flag to allow linebreaks. 
                Default is 'False'.
        Returns:
            str: Extracted string of the uniprot note section.
    """
    if not splitted:
        regex = r"\/note=\"(?P<note>.+?)\""
    else:
        regex = r"\/note=\"(?P<note>.*)"
    result = re.findall(regex, string)
    return result

def extract_note_end(
        string: str, has_mark:bool = True
    ) -> str:
    """
        Helper function to extract information about note of the 
            protein from Uniprot using regular expression.

        Args:
            string (str): Uniprot annotation string.
            has_mark (bool, optional): Flag if end quotation marks are present. 
                Default is 'False'.
        Returns:
            str: Extracted string of the uniprot note section.
    """
    if has_mark:
        regex = r"FT\s+(?P<note>.*)\""
    else:
        regex = r"FT\s+(?P<note>.*)"
    result = re.findall(regex, string)
    return result

def resolve_unclear_position(
        value: str
    ) -> float:
    """
    Replace unclear position of the start/end of the modification 
    defined as '?' with -1 and if it's defined as '?N'
    or '>N' - by removing the '?'/'>'/'<' signs.

    Args:
        value (str): Unclear sequence position from uniprot.
    Returns:
        float: Resolved sequence position.
    """
    # Old implementation commented out:
    # if value == '?': return -1
    # value = value.replace('?', '').replace('>', '').replace('<', '')
    # return float(value)

    if value == '?':
        return -1
    # Remove known non-numeric markers
    cleaned = value.replace('?', '').replace('>', '').replace('<', '')
    # Extract first number (integer or float) from string
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        return float(match.group(1))
    else:
        return np.nan

def extract_positions(
        posit_string: str
    ):
    """
        Extract isoform_id(str) and start/end positions(float) of 
            any feature key from the string.

        Args:
            posit_string (str): Uniprot position string.
        Returns:
            [str, float, float]: 
                str: Uniprot isoform accession, 
                float: start position, 
                float: end position
    """
    isoform = ''
    start = end = np.nan
    if '..' in posit_string:
        start, end = posit_string.split('..')
    if ':' in posit_string:
        if isinstance(start, str):
            isoform, start = start.split(':')
        else:
            isoform, start = posit_string.split(':')
    # in the case when we have only one numeric value as a posit_string
    if isinstance(start, float):
        start = posit_string
    # change the type of start and end into int/float(np.nan)
    if isinstance(start, str):
        start = resolve_unclear_position(start)
    if isinstance(end, str):
        end = resolve_unclear_position(end)
    return isoform, start, end

def preprocess_uniprot(
        path_to_file: str
    ) -> pd.DataFrame:
    """
        A complex complete function to preprocess Uniprot data from 
        specifying the path to a flat text file to the returning a 
        dataframe containing information about:
            - protein_id(str)
            - feature(category)
            - isoform_id(str)
            - start(float)
            - end(float)
            - note information(str)

        Args:
            path_to_file (str): Path to a .txt annotation file 
                directly downloaded from uniprot.
        Returns:
            pd.DataFrame: Dataframe with formatted uniprot annotations

    """
    all_data = []
    with open(path_to_file) as f:

        is_splitted = False
        new_instance = False
        combined_note = []
        line_type = ''

        for line in f:
            if line.startswith(('AC', 'FT')):
                if is_splitted:
                    # in case when the note information is splitted into several lines
                    if line.rstrip().endswith('"'):
                        # if it's the final part of the note
                        combined_note.extend(extract_note_end(line))
                        all_data.append([protein_id, feature, isoform, start, end, " ".join(combined_note)])
                        is_splitted = False
                        new_instance = False
                    else:
                        # if it's the middle part of the note
                        combined_note.extend(extract_note_end(line, has_mark=False))
                elif line.startswith('AC'):
                    # contains the protein_id information
                    if line_type != 'AC':
                        # to prevent a situation when the protein has several AC lines with different names
                        # in this case we are taking the first name in the first line
                        protein_id = line.split()[1].replace(';', '')
                    line_type = 'AC'
                elif line.startswith('FT'):
                    line_type = 'FT'
                    # contains all modifications/preprocessing events/etc., their positions, notes
                    data = line.split()
                    if data[1].isupper() and not data[1].startswith('ECO'):
                            feature = data[1]
                            isoform, start, end = extract_positions(data[2])
                            new_instance = True
                    else:
                        if data[1].startswith('/note'):
                            note = extract_note(line)
                            if note:
                                # if note was created > it contains just one line and can be already added to the data
                                all_data.append([protein_id, feature, isoform, start, end, note[0]])
                                new_instance = False
                            else:
                                # if note is empty > it's splitted into several lines and we create combined_note
                                combined_note = extract_note(line, splitted=True)
                                is_splitted = True
                        else:
                            if new_instance:
                                # in case when we don't have any note but need to add other information about instance
                                all_data.append([protein_id, feature, isoform, start, end, ''])
                                new_instance = False

    # create a dataframe for preprocessed data
    uniprot_df = pd.DataFrame(all_data, columns=['protein_id', 'feature', 'isoform_id', 'start', 'end', 'note'])
    # # change the dtypes of the columns
    # uniprot_df.feature = uniprot_df.feature.astype('category')
    # to filter the instances that don't have a defined start/end position(start=-1 or end=-1)
    uniprot_df = uniprot_df[(uniprot_df.start != -1) & (uniprot_df.end != -1)].reset_index(drop=True)

    return uniprot_df

# ======================================================================================
# Relevant Feature Processing Functions
# ======================================================================================

def process_site_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes SITE feature descriptions to identify, re-label, and standardize
    'Cleavage' and 'Breakpoint' events in a fast, vectorized way.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with feature types and details updated.
    """
    # Create a primary mask to work only on SITE features
    site_mask = df['feature'] == 'SITE'
    if not site_mask.any():
        return df # Return early if no SITE features are present

    # 1. Identify and process Genetic Breakpoints
    breakpoint_mask = site_mask & (df['description'].str.contains('breakpoint for translocation|fusion point', case=False, na=False))
    df.loc[breakpoint_mask, 'feature'] = 'BREAKPOINT'
    df.loc[breakpoint_mask, 'group'] = 'Genetic Breakpoint'
    df.loc[breakpoint_mask, 'agent'] = 'Translocation/Fusion'
    # Extract the specific fusion as the note
    df.loc[breakpoint_mask, 'note'] = df.loc[breakpoint_mask, 'description'].str.extract(r'to form (.+)', expand=False).fillna('Fusion protein formation')

    # 2. Identify and process Proteolytic Cleavage events in order of specificity
    # 2a. Microbial Cleavage
    microbial_mask = site_mask & (df['description'].str.contains(r'\(microbial infection\) cleavage', case=False, na=False))
    df.loc[microbial_mask, 'feature'] = 'CLEAVAGE'
    df.loc[microbial_mask, 'group'] = 'Proteolytic Cleavage'
    df.loc[microbial_mask, 'note'] = 'Microbial Infection'
    df.loc[microbial_mask, 'agent'] = df.loc[microbial_mask, 'description'].str.extract(r'by (.+)', flags=re.IGNORECASE, expand=False).str.strip().str.capitalize()

    # 2b. Standard Cleavage (but not microbial)
    std_cleavage_mask = site_mask & (df['description'].str.contains(r'cleavage; by', case=False, na=False)) & (~microbial_mask)
    df.loc[std_cleavage_mask, 'feature'] = 'CLEAVAGE'
    df.loc[std_cleavage_mask, 'group'] = 'Proteolytic Cleavage'
    df.loc[std_cleavage_mask, 'agent'] = df.loc[std_cleavage_mask, 'description'].str.extract(r'by ([^;]+)', flags=re.IGNORECASE, expand=False).str.strip().str.capitalize()
    df.loc[std_cleavage_mask, 'note'] = df.loc[std_cleavage_mask, 'description'].str.split(';').str[2].str.strip()

    # 2c. Autocatalytic Cleavage
    autolysis_mask = site_mask & (df['description'].str.contains('autolysis|auto-cleavage', case=False, na=False))
    df.loc[autolysis_mask, 'feature'] = 'CLEAVAGE'
    df.loc[autolysis_mask, 'group'] = 'Proteolytic Cleavage'
    df.loc[autolysis_mask, 'agent'] = 'Autolysis'
    df.loc[autolysis_mask, 'note'] = 'Autocatalytic'

    # 2d. Generic Cleavage
    generic_cleavage_mask = site_mask & (df['description'].str.startswith('Cleavage', na=False)) & \
                            ~(microbial_mask | std_cleavage_mask | autolysis_mask) # Ensure it wasn't already caught
    df.loc[generic_cleavage_mask, 'feature'] = 'CLEAVAGE'
    df.loc[generic_cleavage_mask, 'group'] = 'Proteolytic Cleavage'
    df.loc[generic_cleavage_mask, 'agent'] = 'Unspecified'

    # 3. Classify all remaining SITEs as generic 'Functional Site'
    processed_mask = breakpoint_mask | microbial_mask | std_cleavage_mask | autolysis_mask | generic_cleavage_mask
    remaining_sites_mask = site_mask & (~processed_mask)
    df.loc[remaining_sites_mask, 'group'] = 'Functional Site'
    
    return df

def process_lipid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes LIPID feature descriptions in a DataFrame in a fast, vectorized way.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for LIPID features.
    """
    # 1. Create a mask to operate only on LIPID features
    lipid_mask = df['feature'] == 'LIPID'
    if not lipid_mask.any():
        return df # Return early if no LIPID features are present

    # 2. Extract key components using a single, powerful regex
    # CORRECTED a "bad character range" error by escaping the hyphen: \-
    regex_pattern = re.compile(
        r"^(?P<modification>[\w\s\(\)\-'.]+?)\s(?P<residue>cysteine|glycine|serine|lysine|asparagine|aspartate|alanine|glutamate)(?: ester)?(?:;)?(?P<details>.*)?$",
        re.IGNORECASE
    )
    
    extracted_df = df.loc[lipid_mask, 'description'].str.extract(regex_pattern)

    # 3. Vectorized categorization for the 'group' column using np.select
    conditions = [
        extracted_df['modification'].str.contains('palmitoyl', case=False, na=False),
        extracted_df['modification'].str.contains('myristoyl', case=False, na=False),
        extracted_df['modification'].str.contains('gpi-anchor', case=False, na=False),
        extracted_df['modification'].str.contains('geranylgeranyl', case=False, na=False),
        extracted_df['modification'].str.contains('farnesyl', case=False, na=False),
        extracted_df['modification'].str.contains('cholesterol', case=False, na=False),
        extracted_df['modification'].str.contains('stearoyl', case=False, na=False)
    ]
    choices = [
        'Palmitoylation', 'Myristoylation', 'GPI-anchor', 'Geranylgeranylation',
        'Farnesylation', 'Cholesterol modification', 'Stearoylation'
    ]
    # np.select is a fast, vectorized equivalent of an if/elif/else chain
    df.loc[lipid_mask, 'group'] = np.select(conditions, choices, default='Lipidation')
    
    # 4. Extract the 'agent' (enzyme) from the details part
    df.loc[lipid_mask, 'agent'] = extracted_df['details'].str.extract(r'by ([\w\s,and]+)', expand=False).str.strip()

    # 5. Build the 'note' column from remaining details
    # We combine general notes and add a flag for microbial infections
    notes = extracted_df['details'].str.replace(r'by [\w\s,and]+', '', regex=True).str.strip()
    microbial_note = df.loc[lipid_mask, 'description'].str.contains(r'\(Microbial infection\)', case=False, na=False)
    # Combine notes, handling empty cases
    final_notes = np.where(microbial_note, 'Microbial Infection; ' + notes.fillna(''), notes)
    df.loc[lipid_mask, 'note'] = final_notes
    df['note'] = df['note'].str.strip('; ').replace('', np.nan)

    return df

def process_crosslink_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes CROSSLNK feature descriptions in a DataFrame in a fast, vectorized way.
    cross-link, which is often a specific type of post-translational 
    modification like Ubiquitination or SUMOylation. 
    It extracts the partner protein (e.g., Ubiquitin, SUMO1) 
    and other relevant details.
    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for CROSSLNK features.
    """
    # 1. Create a mask to operate only on CROSSLNK features
    crosslink_mask = df['feature'] == 'CROSSLNK'
    if not crosslink_mask.any():
        return df

    # 2. Extract components using a regex with named groups
    # This captures the bond type, residues involved, and details about the partner
    regex_pattern = re.compile(
        r"^(?P<bond_type>[\w\s'-]+?)\s\((?P<residues>[\w\s-]*?)\)(?P<details>.*)?$",
        re.IGNORECASE
    )
    
    extracted_df = df.loc[crosslink_mask, 'description'].str.extract(regex_pattern)

    # 3. Determine the 'group' based on the partner protein found in the details
    # Extract the primary partner protein (e.g., ubiquitin, SUMO, ISG15)
    partner_protein = extracted_df['details'].str.extract(
        r'in (ubiquitin|sumo|isg15|ufm1|nedd8|atg12)', 
        flags=re.IGNORECASE, 
        expand=False
    ).str.upper()

    # Use np.select for fast conditional assignment of the group
    conditions = [
        partner_protein == 'UBIQUITIN',
        partner_protein == 'SUMO',
        partner_protein == 'ISG15',
        partner_protein == 'UFM1',
        partner_protein == 'NEDD8',
        partner_protein == 'ATG12'
    ]
    choices = [
        'Ubiquitination', 'SUMOylation', 'ISGylation', 
        'UFMylation', 'NEDDylation', 'ATGylation'
    ]
    # Default group is the bond type if no specific partner is found
    default_group = extracted_df['bond_type'].str.strip().fillna('Cross-link')
    df.loc[crosslink_mask, 'group'] = np.select(conditions, choices, default=default_group)

    # 4. Extract the 'agent' (enzyme or process)
    df.loc[crosslink_mask, 'agent'] = extracted_df['details'].str.extract(r'by ([\w\s]+)', expand=False).str.strip().str.capitalize()

    # 5. Build the 'note' column from remaining details
    notes = extracted_df['details'].str.replace(r'by [\w\s]+', '', regex=True)
    # Add a flag for microbial infections
    microbial_note = df.loc[crosslink_mask, 'description'].str.contains(r'\(Microbial infection\)', na=False)
    final_notes = np.where(microbial_note, 'Microbial Infection; ' + notes.fillna(''), notes)
    df.loc[crosslink_mask, 'note'] = final_notes
    df['note'] = df['note'].str.strip('; ').replace('', np.nan)

    return df

def process_carbohydrate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes CARBOHYD feature which contains Glycosylation.
    Focuses on extracting the most critical information which is the 
        type of linkage (N-, O-, S-, C-), the type of sugar attached, 
        and any special subtypes or enzymes involved.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' 
            columns populated for CARBOHYD features.
    """
    carb_mask = df['feature'] == 'CARBOHYD'
    if not carb_mask.any():
        return df

    # --- Step 1: Regex Extraction ---
    regex_pattern = re.compile(
        r"^(?P<linkage>[NOCS](?:-beta)?-linked)\s"
        r"\((?P<sugar>[\w\.]+)\)\s"
        r"(?:\((?P<subtype>[\w\s-]+)\)\s)?"
        r"(?P<residue>\w+)"
        r"(?P<details>.*)?$",
        re.IGNORECASE
    )
    extracted_df = df.loc[carb_mask, 'description'].str.extract(regex_pattern)

    # --- Step 2: Assign High-Level 'group' ---
    conditions_group = [
        extracted_df['subtype'].str.contains('glycation', case=False, na=False),
        extracted_df['linkage'].str.startswith('N', na=False),
        extracted_df['linkage'].str.startswith('O', na=False)
    ]
    choices_group = ['Glycation', 'N-linked', 'O-linked']
    df.loc[carb_mask, 'group'] = np.select(conditions_group, choices_group, default='Other Glycosylation')

    # --- Step 3: Compile a detailed 'note' for tooltips ---
    # Start building the note with the specific sugar and subtype
    notes = 'Sugar: ' + extracted_df['sugar'].fillna('N/A')
    notes += np.where(extracted_df['subtype'].notna(), '; Subtype: ' + extracted_df['subtype'].str.strip().str.capitalize(), '')
    
    # Extract and append the agent (enzyme)
    agent = extracted_df['details'].str.extract(r'by ([\w\s,and]+)', expand=False).str.strip().str.capitalize()
    df.loc[carb_mask, 'agent'] = agent
    notes += np.where(agent.notna(), '; Enzyme: ' + agent, '')
    
    # Clean other details and append them
    other_details = extracted_df['details'].str.replace(r';?\s?by [\w\s,and]+', '', regex=True).str.strip('; ')
    notes += np.where(other_details.notna() & (other_details != ''), '; ' + other_details, '')
    
    # Prepend microbial infection status if present
    microbial_note = df.loc[carb_mask, 'description'].str.contains(r'\(Microbial infection\)', na=False)
    final_notes = np.where(microbial_note, 'Microbial Infection; ' + notes, notes)
    df.loc[carb_mask, 'note'] = final_notes

    return df

def process_disulfide_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes DISULFID feature descriptions, assigning a high-level group 
    and compiling detailed notes.
    disulfide bonds primarily fall into a few key categories: 
    whether the bond is within the same protein chain (intrachain) 
    or between different chains (interchain), 
    and whether it has a special functional role (like being redox-active).

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group' and 'note' columns populated for DISULFID features.
    """

    # 1. Create a mask to operate only on DISULFID features
    disulf_mask = df['feature'] == 'DISULFID'
    if not disulf_mask.any():
        return df

    # 2. Assign the 'group' based on the primary bond type
    # We use np.select for a fast, conditional assignment.
    conditions = [
        df.loc[disulf_mask, 'description'].str.contains('Interchain', case=False, na=False),
        df.loc[disulf_mask, 'description'].str.contains('Redox-active', case=False, na=False)
    ]
    choices = ['Interchain', 'Redox-active']
    # If neither condition is met, the default is Intrachain (the most common type)
    df.loc[disulf_mask, 'group'] = np.select(conditions, choices, default='Intrachain')

    # 3. Compile a detailed 'note' column
    # For Interchain bonds, extract the partner description
    interchain_notes = df.loc[disulf_mask, 'description'].str.extract(r'Interchain\s?(\(.*\)|;.*)', expand=False).str.strip('(); ')
    
    # For other types, we can grab any descriptive text
    other_notes = df.loc[disulf_mask, 'description']
    
    # Combine notes, prioritizing the specific interchain partner information
    df.loc[disulf_mask, 'note'] = interchain_notes.fillna(other_notes)
    
    # Clean up notes that might be identical to the new group name
    df.loc[disulf_mask & (df['note'] == df['group']), 'note'] = np.nan
    # Remove the word 'Interchain' from the note to avoid redundancy
    df.loc[disulf_mask, 'note'] = df.loc[disulf_mask, 'note'].str.replace('Interchain', '', case=False).str.strip('(); ')
    df['note'] = df['note'].replace('', np.nan)

    # Agent is not typically applicable for disulfide bonds
    df.loc[disulf_mask, 'agent'] = np.nan

    return df

def process_varseq_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes VAR_SEQ feature descriptions, simplifying them into a structured format
    that describes the type of change and the affected isoforms.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for VAR_SEQ features.
    """
    # 1. Create a mask to operate only on VAR_SEQ features
    varseq_mask = df['feature'] == 'VAR_SEQ'
    if not varseq_mask.any():
        return df

    # 2. Use a regex to parse the description
    regex_pattern = re.compile(
        r"^(?P<change_type>Missing|.+ -> .+)\s\((?P<isoform_details>in isoform .+)\)$",
        re.IGNORECASE
    )
    
    extracted_df = df.loc[varseq_mask, 'description'].str.extract(regex_pattern)

    # 3. Populate the 'group' column: 'Missing' or 'Modified'
    df.loc[varseq_mask, 'group'] = np.where(
        extracted_df['change_type'].str.lower() == 'missing', 
        'Missing', 
        'Modified'
    )

    # 4. Populate the 'agent' column with CLEANED isoform name(s)
    # FINAL CORRECTION: This regex removes all unwanted words globally.
    isoform_names = extracted_df['isoform_details'].str.replace(r'\b(in|isoform|and)\b', '', regex=True, flags=re.IGNORECASE)
    # This second step cleans up any leftover commas and spaces for a perfect format.
    isoform_names = isoform_names.str.replace(r'[\s,]+', ', ', regex=True).str.strip(', ')
    df.loc[varseq_mask, 'agent'] = isoform_names

    # 5. Populate the 'note' column with the simplified summary
    df.loc[varseq_mask, 'note'] = df.loc[varseq_mask, 'group'] + ' (' + extracted_df['isoform_details'] + ')'
    
    return df

def process_mutagen_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes MUTAGEN feature descriptions, categorizing the functional impact
    and parsing the specific amino acid change.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for MUTAGEN features.
    """
    # 1. Create a mask to operate only on MUTAGEN features
    mutagen_mask = df['feature'] == 'MUTAGEN'
    if not mutagen_mask.any():
        return df

    # 2. Parse the description into 'agent' (the change) and 'note' (the consequence)
    # This regex splits the string at the first colon
    regex_pattern = re.compile(r"^(?P<agent>[^:]+):\s*(?P<note>.*)$")
    extracted_df = df.loc[mutagen_mask, 'description'].str.extract(regex_pattern)
    
    # Assign the extracted parts back to the main DataFrame
    df.loc[mutagen_mask, 'agent'] = extracted_df['agent'].str.strip()
    df.loc[mutagen_mask, 'note'] = extracted_df['note'].str.strip()

    # 3. Categorize the 'note' text to create the 'group' column
    # We use np.select for a prioritized, vectorized search for keywords.
    notes_lower = df.loc[mutagen_mask, 'note'].str.lower()
    
    conditions = [
        # Gain of Function
        notes_lower.str.contains(r'increased|enhances|activating mutant|constitutively active', na=False),
        # Complete Loss of Function
        notes_lower.str.contains(r'abolishes|abolished|loss of function|loss of activity|inactive|no activity', na=False),
        # Reduced Function
        notes_lower.str.contains(r'reduces|reduced|decreased|impaired|diminishes|low activity', na=False),
        # Altered Function
        notes_lower.str.contains(r'alters specificity|changes substrate|converts|switch', na=False),
        # No Significant Effect
        notes_lower.str.contains(r'no effect|does not affect|unaffected|similar to wild-type', na=False),
        # Structural Impact (as a fallback)
        notes_lower.str.contains(r'stability|folding|dimerization|oligomerization', na=False)
    ]
    
    choices = [
        'Gain of Function',
        'Loss of Function',
        'Reduced Function',
        'Altered Function',
        'No Significant Effect',
        'Structural Impact'
    ]
    
    # The default is for complex descriptions that don't fit a simple category
    df.loc[mutagen_mask, 'group'] = np.select(conditions, choices, default='Complex/Specific Effect')

    return df

def process_conflict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes CONFLICT feature descriptions, categorizing the type of conflict,
    and parsing the source reference and specific change.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for CONFLICT features.
    """
    # 1. Create a mask to operate only on CONFLICT features
    conflict_mask = df['feature'] == 'CONFLICT'
    if not conflict_mask.any():
        return df

    # 2. Parse the description into 'note' (the change) and 'agent' (the reference)
    # This regex splits the description into the part before and the part inside the parentheses
    regex_pattern = re.compile(r"^(?P<note>.+?)\s+\((?P<agent>in Ref\..*)\)$")
    
    extracted_df = df.loc[conflict_mask, 'description'].str.extract(regex_pattern)
    
    # Assign the extracted parts back to the main DataFrame
    df.loc[conflict_mask, 'note'] = extracted_df['note'].str.strip()
    df.loc[conflict_mask, 'agent'] = extracted_df['agent'].str.replace('in ', '', n=1).str.strip()


    # 3. Categorize the type of change to create the 'group' column
    # We use np.select for a prioritized, vectorized classification
    note_series = df.loc[conflict_mask, 'note']
    
    conditions = [
        note_series.str.contains(r'-> Missing', na=False),
        note_series.str.contains(r'Missing ->', na=False),
        # A single letter sequence -> single letter sequence
        note_series.str.match(r'^[A-Z] -> [A-Z]$', na=False),
    ]
    
    choices = ['Deletion', 'Insertion', 'Substitution']
    
    # The default is for multi-residue changes
    df.loc[conflict_mask, 'group'] = np.select(conditions, choices, default='Complex')

    return df

def process_modres_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes MOD_RES feature descriptions, assigning a high-level biological group
    and compiling detailed notes and agents.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for MOD_RES features.
    """
    # 1. Create a mask to operate only on MOD_RES features
    modres_mask = df['feature'] == 'MOD_RES'
    if not modres_mask.any():
        return df

    # 2. Split the description into the main modification and other details
    # This is more robust than a single complex regex for this feature
    temp_df = df.loc[modres_mask, 'description'].str.split(';', n=1, expand=True)
    main_mod = temp_df[0]
    details = temp_df[1]

    # 3. Create the 'group' column using prioritized keyword matching (np.select)
    main_mod_lower = main_mod.str.lower()
    conditions = [
        main_mod_lower.str.contains('phospho', na=False),
        main_mod_lower.str.contains('acetyl', na=False),
        main_mod_lower.str.contains('methyl', na=False),
        main_mod_lower.str.contains('hydroxy', na=False),
        main_mod_lower.str.contains('adp-ribosyl', na=False),
        main_mod_lower.str.contains(r'succinyl|glutaryl|malonyl|crotonyl|butyryl|propionyl', na=False),
        main_mod_lower.str.contains('ubiquitin', na=False), # Placeholder for future if needed
        main_mod_lower.str.contains('amide', na=False),
        main_mod_lower.str.contains('citrulline', na=False),
        main_mod_lower.str.contains('sulf', na=False),
        main_mod_lower.str.contains(r'nitroso|nitro', na=False),
        main_mod_lower.str.contains('pyrrolidone', na=False)
    ]
    choices = [
        'Phosphorylation', 'Acetylation', 'Methylation', 'Hydroxylation',
        'ADP-ribosylation', 'Acylation', 'Ubiquitination', 'Amidation',
        'Citrullination', 'Sulfation/Sulfonation', 'Nitration/Nitrosylation', 'Pyrrolidone carboxylic acid'
    ]
    df.loc[modres_mask, 'group'] = np.select(conditions, choices, default='Other Modification')
    
    # 4. Extract the 'agent' (enzyme or process) from the details part
    df.loc[modres_mask, 'agent'] = details.str.extract(r'by ([\w\s,/-]+)', expand=False).str.strip()

    # 5. Build the 'note' column from the main modification and other details
    df.loc[modres_mask, 'note'] = main_mod.str.strip()
    
    # Extract other details that are not the agent
    other_details = details.str.replace(r'by [\w\s,/-]+', '', regex=True).str.strip()
    # Append them to the note
    df.loc[modres_mask, 'note'] += np.where(other_details.notna() & (other_details != ''), '; ' + other_details, '')
    
    # Add a flag for microbial infections
    microbial_note = df.loc[modres_mask, 'description'].str.contains(r'\(Microbial infection\)', na=False)
    df.loc[modres_mask, 'note'] = np.where(microbial_note, 'Microbial Infection; ' + df.loc[modres_mask, 'note'], df.loc[modres_mask, 'note'])
    
    return df

def process_variant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes VARIANT feature descriptions, categorizing their impact and parsing
    the specific change and dbSNP identifier.

    Args:
        df: The DataFrame containing UniProt features.

    Returns:
        The DataFrame with 'group', 'agent', and 'note' columns populated for VARIANT features.
    """
    # 1. Create a mask to operate only on VARIANT features
    variant_mask = df['feature'] == 'VARIANT'
    if not variant_mask.any():
        return df

    # 2. Parse the description into 'change' and 'details'
    # CORRECTED REGEX: The hyphen in the character set [A-Z\s->] is now escaped as [A-Z\s\->]
    regex_pattern = re.compile(r"^(?P<change>[A-Z\s\->]+?)(?:\s+\((?P<details>.*)\))?$")
    
    extracted_df = df.loc[variant_mask, 'description'].str.extract(regex_pattern)

    # 3. Categorize the 'group' based on the impact described
    desc_lower = df.loc[variant_mask, 'description'].str.lower()
    conditions = [
        desc_lower.str.contains(r'disease|syndrome|cancer|pathogenic|in patient', na=False),
        desc_lower.str.contains('uncertain significance', na=False),
        desc_lower.str.contains('in dbsnp', na=False)
    ]
    choices = ['Disease-associated', 'Uncertain Significance', 'Polymorphism']
    df.loc[variant_mask, 'group'] = np.select(conditions, choices, default='Unclassified')

    # 4. Extract the 'agent' (dbSNP ID) from the details
    df.loc[variant_mask, 'agent'] = extracted_df['details'].str.extract(r'(rs\d+)', expand=False)
    
    # 5. Build the 'note' column from the change and the text consequence
    # Remove the dbSNP part from the details to avoid redundancy
    consequence = extracted_df['details'].str.replace(r'in dbSNP:rs\d+;?\s?', '', regex=True).str.strip()
    change = extracted_df['change'].str.strip()
    
    # Combine the change and the consequence into a clean note
    df.loc[variant_mask, 'note'] = change + np.where(consequence.notna() & (consequence != ''), '; ' + consequence, '')

    return df

# ======================================================================================
# iPTMnet Data Loading and Processing
# ======================================================================================

def _preprocess_dataframe(df, is_ptm_data=False, verbose=False):
    """Internal helper to clean and standardize columns, optimized for speed."""
    if verbose:
        print(f"Initial shape: {df.shape}")
    df = df.copy()

    if is_ptm_data and 'organism' in df.columns:
        df = df.loc[df['organism'] == 'Homo sapiens (Human)']
        if verbose:
            print(f"Filtered to Homo sapiens, new shape: {df.shape}")

    # Vectorized cleaning and type conversion for key columns
    if 'ptm_type' in df.columns:
        # Converting to category is a major performance boost for subsequent group/merge
        df['ptm_type'] = df['ptm_type'].astype('string').str.upper().fillna('').astype('category')

    if 'site' in df.columns:
        df['site'] = (
            df['site'].astype('string')
            .str.replace(r'^[pP]\.?\s*', '', regex=True, n=1)
            .str.replace(r'\s+', '', regex=True)
            .fillna('')
        )

    # Clean remaining object columns
    for col in df.select_dtypes(include=['object']).columns:
        if col in ('ptm_type', 'site'):
            continue
        df[col] = df[col].astype('string').str.strip().fillna('')

    if verbose:
        print(f"Shape after cleaning: {df.shape}\n")
    return df

def load_and_process_iptmnet(ptm_path, score_path, verbose=False):
    """
    Loads, preprocesses, merges, and finalizes iPTMnet data for analysis.

    This high-performance function integrates PTM and score data, converting key
    columns to categorical types for fast grouping and merging.

    Args:
        ptm_path (str): Path to the ptm.txt file.
        score_path (str): Path to the score.txt file.
        verbose (bool): If True, prints progress and diagnostic information. Defaults to False.

    Returns:
        pd.DataFrame: A fully processed and merged DataFrame.
    """
    # 1. Define File Configurations
    ptm_config = {
        "cols": ['ptm_type', 'source', 'substrate_protein', 'substrate_gene', 'organism', 'site', 'enzyme_protein', 'enzyme_gene', 'note', 'pmid'],
        "dtypes": {'substrate_protein': 'string', 'site': 'string', 'enzyme_protein': 'string', 'ptm_type': 'string', 'organism': 'string'}
    }
    score_config = {
        "cols": ['substrate_protein', 'site', 'enzyme_protein', 'ptm_type', 'score'],
        "dtypes": {'substrate_protein': 'string', 'site': 'string', 'enzyme_protein': 'string', 'ptm_type': 'string'}
    }

    # 2. Load and Preprocess Data
    if verbose:
        print(f"--- Loading and Processing {ptm_path} ---")
    ptm_data = pd.read_csv(ptm_path, sep="\t", engine='pyarrow', header=None, names=ptm_config['cols'], dtype=ptm_config['dtypes'])
    ptm_data = _preprocess_dataframe(ptm_data, is_ptm_data=True, verbose=verbose)

    if verbose:
        print(f"--- Loading and Processing {score_path} ---")
    score_data = pd.read_csv(score_path, sep='\t', engine='pyarrow', header=None, names=score_config['cols'], dtype=score_config['dtypes'])
    score_data = _preprocess_dataframe(score_data, verbose=verbose)

    # 3. Aggregate Scores
    if verbose:
        print("--- Aggregating Scores ---")
    score_agg = (
        score_data
        .groupby(['substrate_protein', 'site', 'ptm_type', 'enzyme_protein'], observed=True, dropna=False, as_index=False)['score']
        .max()
    )
    score_agg['score'] = pd.to_numeric(score_agg['score'], errors='coerce').fillna(0).astype('int32')
    if verbose:
        print(f"Aggregated scores into {score_agg.shape[0]:,} unique events.\n")
    
    # 4. Merge DataFrames and Finalize Columns
    if verbose:
        print("--- Merging DataFrames and Finalizing Columns ---")
    merged_data = ptm_data.merge(
        score_agg,
        on=['substrate_protein', 'site', 'ptm_type', 'enzyme_protein'],
        how='left'
    )
    merged_data['score'] = merged_data['score'].fillna(0).astype('int32')
    if verbose:
        n_total = len(merged_data)
        n_with_score = (merged_data['score'] > 0).sum()
        print(f"Merged scores: {n_with_score:,} / {n_total:,} rows have a non-zero score.")
    
    # Create start/end columns
    start_series = pd.to_numeric(merged_data['site'].str.extract(r'(\d+)', expand=False), errors='coerce')
    merged_data['start'] = start_series.astype('Int64')
    merged_data['end'] = merged_data['start']
    if verbose:
        print("Created 'start' and 'end' columns from 'site'.\n")

    return merged_data

def _standardize_iptmnet(df_iptm: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper function to transform a pre-processed iPTMnet DataFrame 
    into the UniProt feature schema.
    """
    # Define mappings
    feature_map = {
        'PHOSPHORYLATION': 'MOD_RES', 'N-PHOSPHORYLATION': 'MOD_RES', 'ACETYLATION': 'MOD_RES',
        'METHYLATION': 'MOD_RES', 'DIHYDROXYLATION': 'MOD_RES', 'S-NITROSYLATION': 'MOD_RES',
        'CARBOXYLATION': 'MOD_RES',
        'O-GLYCOSYLATION': 'CARBOHYD', 'N-GLYCOSYLATION': 'CARBOHYD', 'C-GLYCOSYLATION': 'CARBOHYD',
        'S-GLYCOSYLATION': 'CARBOHYD',
        'MYRISTOYLATION': 'LIPID', 'FARNESYLATION': 'LIPID', 'PALMITOYLATION': 'LIPID',
        'GERANYLGERANYLATION': 'LIPID',
        'UBIQUITINATION': 'CROSSLNK', 'SUMOYLATION': 'CROSSLNK', 'NEDDYLATION': 'CROSSLNK'
    }

    group_map = {
        'PHOSPHORYLATION': 'Phosphorylation', 'N-PHOSPHORYLATION': 'Phosphorylation',
        'ACETYLATION': 'Acetylation', 'METHYLATION': 'Methylation', 'DIHYDROXYLATION': 'Hydroxylation',
        'S-NITROSYLATION': 'Nitration/Nitrosylation', 'CARBOXYLATION': 'Other Modification',
        'O-GLYCOSYLATION': 'O-linked', 'N-GLYCOSYLATION': 'N-linked', 'C-GLYCOSYLATION': 'C-linked',
        'S-GLYCOSYLATION': 'S-linked',
        'MYRISTOYLATION': 'Lipidation', 'FARNESYLATION': 'Lipidation', 'PALMITOYLATION': 'Lipidation',
        'GERANYLGERANYLATION': 'Lipidation',
        'UBIQUITINATION': 'Ubiquitination/SUMOylation', 'SUMOYLATION': 'Ubiquitination/SUMOylation',
        'NEDDYLATION': 'Ubiquitination/SUMOylation'
    }
    
    df_std = pd.DataFrame()

    # Map and rename columns
    df_std['Protein'] = df_iptm['substrate_protein']
    df_std['feature'] = df_iptm['ptm_type'].map(feature_map)
    df_std['start'] = df_iptm['start']
    df_std['end'] = df_iptm['end']
    df_std['group'] = df_iptm['ptm_type'].map(group_map)
    df_std['agent'] = df_iptm['enzyme_protein']
    df_std['score'] = df_iptm['score']
    
    # Create the new 'note' and 'description' columns
    df_std['note'] = df_std['group'].astype(str) + ' at ' + df_iptm['site'].astype(str)
    
    # CORRECTED LINE: Convert categorical 'ptm_type' to string for concatenation
    description_series = 'Source: ' + df_iptm['source'].astype(str) + '; Type: ' + df_iptm['ptm_type'].astype(str)
    
    # Add PMID and original note only if they exist
    description_series += np.where(df_iptm['pmid'].notna() & (df_iptm['pmid'] != ''), '; PMID:' + df_iptm['pmid'].astype(str), '')
    description_series += np.where(df_iptm['note'].notna() & (df_iptm['note'] != ''), '; Note: ' + df_iptm['note'].astype(str), '')
    df_std['description'] = description_series

    df_std['isoform'] = np.nan
    df_std.dropna(subset=['feature'], inplace=True)
    return df_std

def add_iptmnet_to_uniprot(uniprot_df: pd.DataFrame, iptmnet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes iPTMnet data and concatenates it with UniProt data.
    """
    print("--- Integrating iPTMnet and UniProt Data ---")
    
    # 1. Prepare UniProt data: add a default score
    uniprot_prepared = uniprot_df.copy()
    uniprot_prepared['score'] = 3
    print(f"Added default score of 3 to {len(uniprot_prepared):,} UniProt annotations.")

    # 2. Standardize iPTMnet data
    iptmnet_standardized = _standardize_iptmnet(iptmnet_df)
    print(f"Standardized {len(iptmnet_standardized):,} iPTMnet annotations.")

    # 3. Ensure column consistency and order
    final_columns = ['Protein', 'feature', 'start', 'end', 'group', 'agent', 'note', 'description', 'score', 'isoform']
    uniprot_prepared = uniprot_prepared.reindex(columns=final_columns)
    iptmnet_standardized = iptmnet_standardized.reindex(columns=final_columns)

    # 4. Concatenate
    combined_df = pd.concat([uniprot_prepared, iptmnet_standardized], ignore_index=True)
    print(f"Concatenated data. New total annotations: {len(combined_df):,}")

    # 5. Finalize data types and sort
    combined_df['start'] = pd.to_numeric(combined_df['start'], errors='coerce').astype('Int64')
    combined_df['end'] = pd.to_numeric(combined_df['end'], errors='coerce').astype('Int64')
    combined_df.sort_values(by=['Protein', 'start', 'end'], inplace=True)
    print("Final DataFrame sorted by Protein, start, and end positions.")
    
    return combined_df


# ======================================================================================
# MEROPS Data Loading and Processing
# ======================================================================================

def _decode_bytes(x):
    """Helper function to decode byte strings from the MEROPS file."""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode('utf-8')
        except Exception:
            return x.decode('latin-1', errors='replace')
    if isinstance(x, str) and x.startswith("b'") and x.endswith("'"):
        return x[2:-1]
    return x

def process_merops_data(file_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Loads, cleans, and standardizes MEROPS cleavage data to match the UniProt schema.

    Args:
        file_path (str): The path to the mer.tab file.
        verbose (bool): If True, prints progress information. Defaults to False.

    Returns:
        pd.DataFrame: A standardized DataFrame ready for integration.
    """
    if verbose:
        print(f"--- Processing MEROPS data from {file_path} ---")

    # 1. Load the raw data
    mer_data = pd.read_csv(file_path, sep="\t", engine='pyarrow')

    # 2. Decode byte-string columns
    obj_cols = mer_data.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # Check if decoding is needed before applying the map function for performance
        if mer_data[col].map(lambda v: isinstance(v, (bytes, bytearray)) or (isinstance(v, str) and v.startswith("b'"))).any():
            mer_data[col] = mer_data[col].map(_decode_bytes)

    # 3. Filter for human substrates and select relevant columns
    mer_data = mer_data[mer_data['Substrate organism'].isin(['Homo sapiens', 'Homo sapiens (Human)'])].copy()
    mer_data = mer_data[[
        'Substrate (Uniprot)', 'Substrate name', 'Residue number', 
        'Cleaved residue', 'Cleavage type', 'Protease (MEROPS)', 'Protease name'
    ]]

    # 4. Create the new standardized DataFrame according to our design
    df_std = pd.DataFrame()
    df_std['Protein'] = mer_data['Substrate (Uniprot)']
    df_std['feature'] = 'CLEAVAGE'
    df_std['start'] = pd.to_numeric(mer_data['Residue number'], errors='coerce').astype('Int64')
    df_std['end'] = df_std['start']
    df_std['group'] = 'Proteolytic Cleavage'
    df_std['agent'] = mer_data['Protease name']
    df_std['score'] = 2
    df_std['isoform'] = np.nan

    # 5. Construct the 'note' and 'description' columns
    cleavage_map = {
        'non-physiological': 'N-Ph', 'physiological': 'Ph', 'theoretical': 'Th',
        'synthetic': 'Syn', 'unknown': 'Unk', 'pathological': 'Pth'
    }
    cleavage_code = mer_data['Cleavage type'].map(cleavage_map).fillna('Unk')
    
    # Simple summary for the 'note' column
    df_std['note'] = (
        cleavage_code + ' cleavage at ' + 
        mer_data['Cleaved residue'].astype(str) + 
        mer_data['Residue number'].astype(str) + 
        ' by ' + mer_data['Protease name'].astype(str)
    )
    
    # Detailed metadata for the 'description' column
    df_std['description'] = (
        'MEROPS Cleavage Site; Protease: ' + mer_data['Protease name'].astype(str) + 
        ' (MEROPS:' + mer_data['Protease (MEROPS)'].astype(str) + '); Substrate: ' + 
        mer_data['Substrate name'].astype(str)
    )

    # 6. Finalize and reorder columns to match the target schema
    final_columns = [
        'Protein', 'feature', 'start', 'end', 'group', 'agent', 
        'note', 'description', 'score', 'isoform'
    ]
    df_std = df_std[final_columns].dropna(subset=['Protein', 'start']).reset_index(drop=True)

    if verbose:
        print(f"Processed {len(df_std):,} MEROPS cleavage sites for Homo sapiens.")
        
    return df_std

def add_merops_to_uniprot(uniprot_df: pd.DataFrame, merops_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Concatenates standardized MEROPS data with the main UniProt DataFrame.

    Args:
        uniprot_df: The main DataFrame containing parsed UniProt and other features.
        merops_df: The standardized DataFrame from process_merops_data().
        verbose (bool): If True, prints progress information. Defaults to False.

    Returns:
        A single, combined DataFrame containing all features, sorted by
        Protein, start, and end positions.
    """
    if verbose:
        print("\n--- Adding MEROPS data to UniProt DataFrame ---")

    # Ensure both dataframes have the exact same column structure before combining
    final_columns = [
        'isoform', 'Protein', 'feature', 'start', 'end', 
        'group', 'agent', 'note', 'description', 'score'
    ]
    
    # Make copies to avoid modifying original dataframes
    uniprot_prepared = uniprot_df.copy().reindex(columns=final_columns)
    merops_prepared = merops_df.copy().reindex(columns=final_columns)
    
    # Concatenate the two dataframes
    combined_df = pd.concat([uniprot_prepared, merops_prepared], ignore_index=True)
    
    if verbose:
        print(f"Added {len(merops_prepared):,} MEROPS annotations. New total: {len(combined_df):,}")

    # Finalize data types and sort
    combined_df['start'] = pd.to_numeric(combined_df['start'], errors='coerce').astype('Int64')
    combined_df['end'] = pd.to_numeric(combined_df['end'], errors='coerce').astype('Int64')
    # Replace NaNs with empty strings if string columns
    str_cols = ['isoform', 'Protein', 'feature', 'group', 'agent', 'note', 'description']
    for col in str_cols:
        combined_df[col] = combined_df[col].astype('string').fillna('')
    
    combined_df.sort_values(by=['Protein', 'start', 'end'], inplace=True, na_position='first')
    
    if verbose:
        print("Final DataFrame sorted by Protein, start, and end positions.")
        
    return combined_df

def add_feature_metadata(df, feature_categories, relevant_features):
    """
    Adds 'is_relevant' and 'feature_category' columns to the dataframe.

    Args:
        df (pd.DataFrame): The main dataframe with a 'feature' column.
        feature_categories (dict): Dictionary mapping categories to lists of features.
        relevant_features (list): List of features considered relevant.

    Returns:
        pd.DataFrame: The dataframe with two new metadata columns.
    """
    df = df.copy()

    # 1. Add the 'is_relevant' column (boolean)
    df['is_relevant'] = df['feature'].isin(relevant_features)

    # 2. Add the 'feature_category' column (string)
    # Create a reversed map for efficient lookup (feature -> category)
    feature_to_category_map = {
        feature: category
        for category, features in feature_categories.items()
        for feature in features
    }
    df['feature_category'] = df['feature'].map(feature_to_category_map)

    return df

# ======================================================================================
# Main Pipeline Function
# ======================================================================================