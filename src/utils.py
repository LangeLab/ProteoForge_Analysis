import re
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import matthews_corrcoef
from sknetwork.hierarchy import cut_balanced

from dynamicTreeCut import cutreeHybrid
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

from Bio import SeqIO
from Bio.SeqUtils import molecular_weight

##################### Global Variable Utilized #####################
# TODO: Add the global variables as needed
# Cell
uniprot_feature_dict = {
    'CHAIN': 'Chain',
    'INIT_MET': 'Initiator methionine',
    'PEPTIDE': 'Peptide',
    'PROPEP': 'Propeptide',
    'SIGNAL': 'Signal peptide',
    'TRANSIT': 'Transit peptide',
    'CROSSLNK': 'Cross-link',
    'DISULFID': 'Disulfide bond',
    'CARBOHYD': 'Glycosylation',
    'LIPID': 'Lipidation',
    'MOD_RES': 'Modified residue',
    'COILED': 'Coiled coil',
    'COMPBIAS': 'Compositional bias',
    'DOMAIN': 'Domain',
    'MOTIF': 'Motif',
    'REGION': 'Region',
    'REPEAT': 'Repeat',
    'ZN_FING': 'Zinc finger',
    'INTRAMEM': 'Intramembrane',
    'TOPO_DOM': 'Topological domain',
    'TRANSMEM': 'Transmembrane',
    'STRAND': 'Beta strand',
    'HELIX': 'Helix',
    'TURN': 'Turn',
    'ACT_SITE': 'Active site',
    'BINDING': 'Binding site',
    'CA_BIND': 'Calcium binding',
    'DNA_BIND': 'DNA binding',
    'METAL': 'Metal binding',
    'NP_BIND': 'Nucleotide binding',
    'SITE': 'Site',
    'NON_STD': 'Non-standard residue',
    'NON_CONS': 'Non-adjacent residues',
    'NON_TER': 'Non-terminal residue',
    'VARIANT': 'Natural variant',
    'CONFLICT': 'Sequence conflict',
    'VAR_SEQ': 'Alternative sequence',
    'UNSURE': 'Sequence uncertainty',
    'STRUCTURE': 'Secondary structure',
    'MUTAGEN': 'Mutagenesis'
}

#################### Notebook Utility functions ####################

def getTime() -> float:
    """
        Get the current time for timer

        Returns:
            float: The current time in seconds.
    """
    return time.time()

def prettyTimer(
        seconds: float
    ) -> str:
    """
        Better way to show elapsed time

        Args:
            seconds (float): The number of seconds to convert to a pretty format.

        Returns:
            str: The elapsed time in a pretty format.
        
        Examples:
            >>> prettyTimer(100)
            '00h:01m:40s'

            >>> prettyTimer(1000)
            '00h:16m:40s'

            >>> prettyTimer(10000)
            '02h:46m:40s'
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02dh:%02dm:%02ds" % (h, m, s)

def view_table(
        data: pd.DataFrame, 
        page_size: int = 25, 
        page_number: int = None
    ):
    """
        Displays a table with pages with current page or 
            with all pages if not specified. 
            Going around the VScodes view limit.
    
    """
    from IPython.display import display
    n_pages = len(data) // page_size + 1
    if page_number is not None:
        print(f"Total pages: {n_pages}, Current page: {page_number}")
        if page_number < 1 or page_number > n_pages:
            print(f"Invalid page number. Please select a page between 1 and {n_pages}.")
        else:
            display(data.iloc[(page_number - 1) * page_size:page_number * page_size])
    else:
        print(f"Total pages: {n_pages}")
        for i in range(n_pages):
            display(data.iloc[i * page_size:(i + 1) * page_size])

def print_shape(
        df: pd.DataFrame, 
        identifier: str ="", 
        behavior: str ="print"
    ) -> None:
    """
        Print the shape of a pandas dataframe.

        Args:
            df (pd.DataFrame): The pandas dataframe to print.
            identifier (str, optional): The identifier to print. Defaults to "".
            behavior (str, optional): The behavior of the function. 
                "print" to print the shape, "return" to return the shape. Defaults to "print".  
        
        Raises:
            TypeError: If df is not a pandas dataframe.
            TypeError: If identifier is not a string.
            ValueError: If behavior is not "print" or "return".

        Examples:
            >>> print_shape(pd.DataFrame(), "My Data", "print")
            My Data data has 0 rows and 0 columns

            >>> print_shape(pd.DataFrame(), "My Data", "return")
            'My Data data has 0 rows and 0 columns'

            >>> print_shape(pd.DataFrame(), "My Data", "invalid")
            ValueError: behavior must be either "print" or "return"
    """
    if behavior == "print":
        print(f"{identifier} data has {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    elif behavior == "return":
        return f"{identifier} data has {df.shape[0]:,} rows and {df.shape[1]:,} columns"
    else:
        raise ValueError("behavior must be either 'print' or 'return'")

def print_series(
        series: pd.Series, 
        header: str = None, 
        tab: int = 0,
        elements_with_order: list = None
    ) -> None:
    """
        Print a pandas series with an optional header

        Args:
            series (pd.Series): The pandas series to print.
            header (str, optional): The header to print. Defaults to None.
            tab (int, optional): The number of spaces to print before each element. Defaults to 0.
            elements_with_order (list, optional): A list of elements to print. Defaults to None.

        Raises:
            TypeError: If series is not a pandas series.
            TypeError: If header is not a string.
            TypeError: If tab is not an integer.
            TypeError: If elements_with_order is not a list.
            ValueError: If tab is less than 0.

        Examples:
            >>> print_series(pd.Series([1, 2, 3]), "My Series", 4, ["a", "b", "c"])
            My Series
                a -> 1
                b -> 2
                c -> 3

            >>> print_series(pd.Series([1, 2, 3]), "My Series", 4)
            My Series
                0 -> 1
                1 -> 2
                2 -> 3

            >>> print_series(pd.Series([1, 2, 3]))
                0 -> 1
                1 -> 2
                2 -> 3
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas series")
    if not isinstance(header, str) and header is not None:
        raise TypeError("header must be a string")
    if not isinstance(tab, int):
        raise TypeError("tab must be an integer")
    if not isinstance(elements_with_order, list) and elements_with_order is not None:
        raise TypeError("elements_with_order must be a list")
    if tab < 0:
        raise ValueError(
            """
            tab must be a positive integer amount.Indicating the empty space prior to printing each element
            """
        )

    if header is not None:
        print(header)
    if elements_with_order is not None:
        for i in elements_with_order:
            if i in series.index:
                print(" "*tab, i, "->", series[i])
    else:
        for index, value in series.items():
                print(" "*tab, index, "->", value)

def print_list(
        data: list,
        n_elements: 3,
        description: str = "Preview of the list: ",
    ):
    """ 
        Print a preview of a list with a description and a specified 
        number of elements to show from the start and end of the list.
    """
    # Print the description
    print(
        f"{description} {data[:n_elements]}...{data[-n_elements:]}"
    )

#################### Fasta file parsing functions ####################

def getMW(
        seq: str, 
        kDa: bool = True
    ) -> float:
    """
        Get the molecular weight of a protein sequence.

        Args:
            seq (str): Amino acid sequence.
            kDa (bool, optional): If True, the molecular weight is returned in kDa. Defaults to True.

        Returns:
            float: The molecular weight of the sequence in kDa if kDa is True, else in Da.
                If the sequence contains invalid amino acids, np.nan is returned.

        Examples:
            >>> getMW("ACDEFGHIKLMNPQRSTVWY")
            2.3

            >>> getMW("ACDEFGHIKLMNPQRSTVWYX")
            np.nan

            >>> getMW("ACDEFGHIKLMNPQRSTVWYB")
            np.nan

    """
    multiplier = 1
    if kDa: multiplier = 1000

    try: 
        return molecular_weight(seq, "protein") / multiplier
    except ValueError:
        # error thrown, for example, when dealing w/ X amino acid
        return np.nan
    
def remove_sequences_with_invalid_AA(
        seq: str
    ) -> str:
    """
        Remove sequences with invalid amino acids.

        Args:
            seq (str): Amino acid sequence.

        Returns:
            str: The amino acid sequence if it contains no invalid amino acids, else np.nan.
        
        Examples:
            >>> remove_sequences_with_invalid_AA("ACDEFGHIKLMNPQRSTVWY")
            "ACDEFGHIKLMNPQRSTVWY"

            >>> remove_sequences_with_invalid_AA("ACDEFGHIKLMNPQRSTVWYX")
            np.nan

            >>> remove_sequences_with_invalid_AA("ACDEFGHIKLMNPQRSTVWYB")
            np.nan
    """
    # Invalid characters are: X, B, Z, J, U, O, *
    if re.search("[X|B|Z|J|U|O|\*]", seq):
        return np.nan
    else:
        return seq

def check_length(
        seq: str,
        minL: int = 7,
        maxL: int = 50 
    ) -> bool:
    """
        Check if a peptide/protein sequence is within a specified length range.

        Args:
            seq (str): A peptide or protein sequence.
            minL (int, optional): The minimum length of the sequence. Defaults to 7.
            maxL (int, optional): The maximum length of the sequence. Defaults to 50.
                For proteins should be set to 10**6 to avoid any length restrictions.

        Returns:
            bool: True if the peptide sequence is within the length range, else False.

        Examples:
            >>> check_length("ACDEFGHIKLMNPQRSTVWY", minL=7, maxL=50)
            True
            
            >>> check_length("x"*51, maxL=50)
            False

            >>> check_length("x"*6, minL=7)
            False

            >>> check_length("x"*51, minL=7, maxL=10**6)
            True
    """
    return len(seq) >= minL and len(seq) <= maxL

def parse_proteome_header(
        header: str
    ) -> dict:
    """
        Parse a Uniprot fasta proteome header and return a dictionary of informative variables.

        Args:
            header (str): Uniprot fasta proteome header string.

        Returns:
            dict: A dictionary of informative variables.

        Examples:
            >>> parse_proteome_header("sp|Q9Y5Y9|1433B_HUMAN 14-3-3 protein beta GN= YWHAB PE=1 SV=2")
            {'reviewStatus': 'sp', 'entry': 'Q9Y5Y9', 'entryName': '1433B_HUMAN', 
            'geneName': 'YWHAB', 'proteinDescription': '14-3-3 protein beta'}

    """
    # Define regular expressions for each field we want to extract
    regexes = {
        "reviewStatus": r"^(sp|tr)\|([A-Z0-9]+)\|",
        "entry": r"^.*?\|([A-Z0-9-]+)\|",
        "entryName": r"^.*?\|([A-Z0-9]+)_([A-Z]+)",
        "geneName": r" GN=([^ ]+)",
        "proteinDescription": r" ([^=]+)(?<! [A-Z]{2}=)",
    }

    # Extract the information using regular expressions
    variables = {}
    for key, regex in regexes.items():
        match = re.search(regex, header)
        if match:
            if key == "entryName":
                variables[key] = f"{match.group(1)}_{match.group(2)}" if match.group(2) else match.group(1)
            else:
                variables[key] = match.group(1) if key == "proteinDescription" else match.group(1)
                if key == "proteinDescription":
                    variables[key] = variables[key].strip(" OS")

    return variables

# Function to read and parse the fasta file into a dataframe
def fasta_to_df(
        reference_path: str, 
        fasta_ID: str = None,
        geneOnly: bool = False, 
        seqLenMin: int = 7, 
        seqLenMax: int = 10**6,
        col_order: list = [
            "fastaId", "entry", "entryName", 
            "geneName", "proteinDescription",
            "reviewStatus", "isoformStatus", 
            "sequenceLength", "molecularWeight_kDa", 
            "sequence"
        ],
        sort_by: list = ["entry", "isoformStatus"],
        sort_order: list = [True, False]
    ) -> pd.DataFrame:
    """
        Read and parse the Uniprot fasta proteome file into a dataframe.
        
        Args:
            reference_path (str): The path to the Uniprot fasta proteome file.
            fasta_ID (str, optional): The ID of the fasta file. Defaults to None.
            geneOnly (bool, optional): If True, only sequences with gene names are included. Defaults to False.
            seqLenMin (int, optional): The minimum length of the sequence. Defaults to 7.
            seqLenMax (int, optional): The maximum length of the sequence. Defaults to 10**6.
            col_order (list, optional): The order of the columns in the dataframe. Defaults to None.
            sort_by (list, optional): The columns to sort the dataframe by. Defaults to ["entry", "isoformStatus"].
            sort_order (list, optional): The order to sort the dataframe by. Defaults to [True, False].

        Returns:
            pd.DataFrame: A dataframe containing the parsed information from the fasta file.

        Examples:
            >>> fasta_to_df("data/uniprot-proteome_UP000005640.fasta")

            >>> fasta_to_df("data/uniprot-proteome_UP000005640.fasta", "UP000005640")

            >>> fasta_to_df("data/uniprot-proteome_UP000005640.fasta", "UP000005640", geneOnly=True)

            >>> fasta_to_df("data/uniprot-proteome_UP000005640.fasta", "UP000005640", 
            geneOnly=True, seqLenMin=7, seqLenMax=10**6)
    """

    # Initialize a list to hold all the entries
    results = []
    for record in SeqIO.parse(reference_path, "fasta"):
        # Remove sequences with invalid amino acids
        cur_seq = remove_sequences_with_invalid_AA(str(record.seq))
        if cur_seq is np.nan: continue # Skip sequences if with invalid amino acids
        # Skip sequences with length less than 7 or more than 50K
        if not check_length(cur_seq, seqLenMin, seqLenMax): continue
        # Parse the header
        cur_dict = parse_proteome_header(record.description)
        # Skip sequence if gene name is not available 
        if geneOnly and "geneName" not in cur_dict: continue
        # Get the sequence
        cur_dict["sequence"] = cur_seq
        # Add the sequence length
        cur_dict["sequenceLength"] = len(cur_seq)
        # Add molecular weight
        cur_dict["molecularWeight_kDa"] = getMW(cur_seq)
        # Add the entry to the list
        results.append(cur_dict)

    # Convert the list to a dataframe
    df = pd.DataFrame(results)
    # Add the fasta ID
    if fasta_ID is not None: df["fastaId"] = fasta_ID
    # Update the review status
    df["reviewStatus"] = df["reviewStatus"].apply(lambda x: "reviewed" if x == "sp" else "unreviewed") 
    # Update the isoform status
    # If the entry name contains a dash, it is an isoform
    df["isoformStatus"] = df["entry"].apply(lambda x: "isoform" if "-" in x else "canonical")
    # This is not perfect, but it is the best we can do with the information we have 
    # Order the columns
    if col_order is not None:
        if not set(col_order).issubset(df.columns):
            raise ValueError("The col_order list must contain all the columns in the dataframe.")
        df = df[col_order]
    # Sort the dataframe
    if sort_by is not None:
        if not set(sort_by).issubset(df.columns):
            raise ValueError("The sort_by list must contain all the columns in the dataframe.")
        df = df.sort_values(by=sort_by, ascending=sort_order)
    # Return the dataframe
    return df.reset_index(drop=True)

#################### Peptide Info Expansion functions ####################

def select_representative_protein(
        proteins: str
    ) -> str:
    """
        Selects a representative protein from a group

        The function takes a string of protein IDs separated by semicolons. 
        If there's only one ID, it is returned. If there are multiple IDs, 
        the function prioritizes 6-letter IDs over 10-letter ones. If no 6-letter 
        IDs are present, the first ID is returned.

        Args:
            proteins (str): A string of protein IDs separated by semicolons.

        Returns:
            str: The ID of the representative protein.

        Examples:
            >>> select_representative_protein("P12345")
            'P12345'

            >>> select_representative_protein("A0A075B6K5;Q12345")
            'Q12345'

            >>> select_representative_protein("P12345;A0A075B6K5;P1234")
            'P12345'
    """
    protein_ids = proteins.split(";")
    
    if len(protein_ids) == 1:
        return protein_ids[0]
    
    six_letter_ids = [id for id in protein_ids if len(id) == 6]
    
    return six_letter_ids[0] if six_letter_ids else protein_ids[0]

# Utility function that uses the dictionary from trace builder function into a dataframe
def dict_to_protein_peptide_df(
        data_dict: dict
    ) -> pd.DataFrame:
    """
        Converts a nested dictionary of protein-peptide counts into a Pandas DataFrame with a single 'Trace' column.

        Args:
            data_dict: A dictionary where keys are protein IDs and values are dictionaries of peptide counts.

        Returns:
            pd.DataFrame: A DataFrame with 'Protein', 'Peptide', and 'Trace' columns.
    """
    # Initialize a list to hold the rows
    rows = []
    # Iterate over the dictionary
    for protein, peptide_counts in data_dict.items():
        for peptide, count in peptide_counts.items():
            row = {'Protein': protein, 'Peptide': peptide, 'Trace': count}
            rows.append(row)

    return pd.DataFrame(rows)

# TODO: Add the Protein Coverage and Peptide Tracer functions

# TODO: Function to expand the peptide information (match positions, etc.)

# Function to find overlapping peptides
def group_miss_cleaved_peptides(
        startpos_arr: list,
        endpos_arr: list, 
        max_diff: int = 3
    ) -> dict:
    """
        Group overlapping peptides based on their start and end positions 
            with a maximum allowed difference.

        Args:
            startpos_arr (list): List of peptide start positions.
            endpos_arr (list): List of peptide end positions.
            max_diff (int, optional): The maximum allowed difference between 
                peptide start and end positions. Defaults to 3.

        Returns:
            dict: A dictionary where keys are the indices of the 
                longest peptides and values are lists of overlapping peptide indices.
    """

    overlapping_peptides = {}
    n = len(startpos_arr)
    used_indices = set()
    
    for i in range(n):
        if i in used_indices:
            continue
        
        longest_peptide_index = i
        longest_peptide_length = endpos_arr[i] - startpos_arr[i]
        overlapping_indices = [i]
        
        for j in range(n):
            if i != j and j not in used_indices:
                start1, end1 = startpos_arr[i], endpos_arr[i]
                start2, end2 = startpos_arr[j], endpos_arr[j]
                
                # Check if peptides overlap within the allowed difference
                if (abs(start1 - start2) <= max_diff and abs(end1 - end2) <= max_diff):
                    overlapping_indices.append(j)
                    used_indices.add(j)
                    
                    # Update the longest peptide index if necessary
                    current_peptide_length = endpos_arr[j] - startpos_arr[j]
                    if current_peptide_length > longest_peptide_length:
                        longest_peptide_index = j
                        longest_peptide_length = current_peptide_length
        
        if len(overlapping_indices) > 1:
            overlapping_peptides[longest_peptide_index] = overlapping_indices
            used_indices.update(overlapping_indices)
        else:
            overlapping_peptides[i] = []

    # Add non-grouped peptides with empty lists
    for i in range(n):
        if i not in overlapping_peptides and i not in used_indices:
            overlapping_peptides[i] = []
    
    return overlapping_peptides

################### Uniprot Annotation Functions ###################
# Source: Alphamap/alphamap/uniprot_integration.py
# Main and helper functions used as is without any modifications
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

# Cell
def resolve_unclear_position(
        value: str
    ) -> float:
    """
        Replace unclear position of the start/end of the modification 
            defined as '?' with -1 and if it's defined as '?N'
            or ">N" - by removing the '?'/'>'/'<' signs.

        Args:
            value (str): Unclear sequence position from uniprot.
        Returns:
            float: Resolved sequence position.
    """
    # if it's "1..?" or "?..345" for start or end -> remove -1 that we can filter later
    # if it's "31..?327" or "?31..327" -> remove the question mark
    # if it's "<1..106" or "22..>115" -> remove the "<" or ">" signs
    if value == '?': return -1
    value = value.replace('?', '').replace('>', '').replace('<', '')
    return float(value)

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
    # change the dtypes of the columns
    uniprot_df.feature = uniprot_df.feature.astype('category')
    # to filter the instances that don't have a defined start/end position(start=-1 or end=-1)
    uniprot_df = uniprot_df[(uniprot_df.start != -1) & (uniprot_df.end != -1)].reset_index(drop=True)

    return uniprot_df

#################### Sequence parsing functions ####################

def build_plot_dict(
        protein: str,                   # Protein Name
        quan_data: pd.DataFrame,        # Quantification in the form of results_df in long-format
        info_data: pd.DataFrame,        # Expanded Info Data where compressed info
        uniprot_data: pd.DataFrame,     # Uniprot Annotation Data
        condition_pal: dict,            # Condition: Color dictionary
        cluster_pal: dict,              # Cluster: Color dictionary
        hue_pal: dict,                  # Hue: Color dictionary (must match the unique values in the hue_col)
        
        ## Additional Variables
        hue_col: str = "TumorRegulation", # Column to use for hue in plots
        pThr: float = 0.0001,           # Threshold for p-value to consider significant
        corrMethod: str = "kendall",    # Correlation Method for heatmap [kendall, spearman, pearson] (should be same as used in ProteoForge)
        distanceMethod: str = "euclidean", 
        linkageMethod: str = "complete",

        # Misc
        verbose: bool = False
    ):
    """
        A function to build a large dictionary with all data and related variables 
            for a given protein for plotting and reporting a protein pdf summary. 
        The function will return a dictionary with all the required data for plotting.

        Args:
            protein (str): The protein name to build the plot dictionary for.
            quan_data (pd.DataFrame): The quantification data in long format.
            info_data (pd.DataFrame): The expanded information data.
            uniprot_data (pd.DataFrame): The uniprot annotation data.
            condition_pal (dict): The condition color dictionary.
            cluster_pal (dict): The cluster color dictionary.
            hue_pal (dict): The hue color dictionary.
            hue_col (str, optional): The column to use for hue in plots. Defaults to "TumorRegulation".
            pThr (float, optional): The threshold for p-value to consider significant. Defaults to 0.0001.
            corrMethod (str, optional): The correlation method for heatmap. Defaults to "kendall".
            distanceMethod (str, optional): The distance method for clustering. Defaults to "euclidean".
            linkageMethod (str, optional): The linkage method for clustering. Defaults to "complete".
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        
        Returns:
            dict: A dictionary with all the required data for plotting.
                    
    """
    ## Global Variables
    info_cols = [
        "Protein", "Peptide", "PeptideID", "proteinDescription", "geneName",
        "startpos", "endpos", "trace", "seqLength", "Cov%", "isCAP", hue_col, 
        "cluster_id", "adj.pvalue", "isSignificant", "ProteoformGroup",
    ]
    quan_cols = [
        "Protein", "Peptide", "Condition", "Sample",
        "Intensity", "log10(Intensity)", "adjIntensity", 
        "isReal", "isCompMiss", "Weight", 
    ]
    uniprot_cols = ["protein_id", "start", "end", "feature", "note"]

    ## Validations
    
    assert all([
        col in info_data.columns for col in info_cols
    ]), "Missing required columns in info_data"
    assert all([
        col in quan_data.columns for col in quan_cols
    ]), "Missing required columns in quan_data"
    assert all([
        col in uniprot_data.columns for col in uniprot_cols
    ]), "Missing required columns in uniprot_data"
    

    ## Subset data
    # Information Retrieval
    cur_info = info_data[
        info_data['Protein'] == protein
    ]
    # Annotation Retrieval
    cur_annot = uniprot_data[
        uniprot_data["protein_id"] == protein
    ].sort_values(["start", "end"])
    # Quantification Retrieval
    cur_quan = quan_data.loc[
        quan_data["Protein"] == protein,
        quan_cols
    ]
    # Merge the information
    plot_data = cur_quan.merge(
        cur_info,
        on=["Protein", "Peptide"],
        how="left"
    ) 
    plot_data["-log10(adj.pvalue)"] = -np.log10(plot_data["adj.pvalue"])

    ## Gather details
    NumPeptides = cur_info.shape[0]
    NumSignfPeptides = cur_info["isSignificant"].sum()
    geneName = cur_info["geneName"].values[0]
    proteinLength = cur_info["seqLength"].values[0]
    proteinCov = cur_info["Cov%"].values[0]

    ## Correlation details
    # correlation annotation
    corrAnnot = plot_data[[
        # TODO: Add more columns for complexHeatmap Annotations
        "PeptideID", '-log10(adj.pvalue)', 'isSignificant',
        hue_col, 'cluster_id', "ProteoformGroup"
    ]].drop_duplicates().set_index("PeptideID").rename(columns={
        # '-log10(adj.pvalue)': "PValue",
        'cluster_id': "Cluster",
    })
    corrAnnot["Annot"] = corrAnnot.index
    corrAnnot.loc[corrAnnot["isSignificant"] == 0, "Annot"] = np.nan
    corrAnnot['PFGroup'] = "PFG_" + corrAnnot['ProteoformGroup'].astype('category').cat.codes.astype(str)
    # correlation data
    corr_matrix = plot_data.pivot(
        index="PeptideID",
        columns="Sample",
        values="Intensity"
    )
    distance_matrix = 1 - corr_matrix
    corr_matrix = distance_matrix.T.corr(method=corrMethod).stack()
    corr_matrix.index.names = ["level_0", "level_1"]
    corr_matrix = corr_matrix.reset_index(name="Correlation")

    tmp = cur_info[cur_info["isSignificant"]]
    # Assemble Information
    plot_dict = {
        "Protein": protein,
        "ProteinName": cur_info["proteinDescription"].unique()[0],
        "Gene": geneName,
        "pThr": pThr,
        "Peptides": tmp["Peptide"].to_list(),
        "p_values": tmp["adj.pvalue"].to_list(),
        "PeptideIds": tmp["PeptideID"].to_list(),
        "ProteinLength": proteinLength,
        "ProteinCoverage": proteinCov,
        "ConditionColors": condition_pal,
        "HueColors": hue_pal,
        "ClusterColors": cluster_pal,
        # Cluster Information
        "distanceMethod": distanceMethod,
        "linkageMethod": linkageMethod,
        # Various Tables
        "ResultSubset": cur_info,
        "UniprotAnnotData": cur_annot,
        "PlotData": plot_data,
        "CorrData": corr_matrix,
        "CorrAnnot": corrAnnot,
    }

    if verbose:
        print(f"{geneName} ({protein}) - {proteinLength:.0f} aa - {proteinCov:.2f}% Coverage")
        print(f"# of Peptides (All/Sgnf): {NumPeptides}/{NumSignfPeptides}")

    return plot_dict

#################### Quantitative calculation functions ####################

def cv_numpy(
        x: np.ndarray, 
        axis: int = 1, 
        ddof: int = 1, 
        ignore_nan: bool = False, 
        format: str = "percent"
    ) -> np.ndarray:
    """
        Calculates the coefficient of variation of the values in the passed array.
        
        Args:
            x (np.ndarray): The array of values.
            axis (int, optional): The axis along which to calculate the coefficient of variation. Defaults to 1.
            ddof (int, optional): The degrees of freedom. Defaults to 1.
            ignore_nan (bool, optional): If True, NaN values are ignored. Defaults to False.
            format (str, optional): The format of the output. "percent" for percentage, "ratio" for ratio. 
                Defaults to "percent".

        Returns:
            np.ndarray: The coefficient of variation of the values in the passed array.

        Raises:
            TypeError: If x is not a numpy array.
            TypeError: If axis is not an integer.
            TypeError: If ddof is not an integer.

        Examples:
            >>> cv_numpy(np.array([1, 2, 3, 4, 5]))
            47.14045207910317

            >>> cv_numpy(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
            array([47.14045208, 47.14045208, 47.14045208])

            >>> cv_numpy(np.array([1, 2, 3, 4, 5]), ignore_nan=True)
            47.14045207910317

            >>> cv_numpy(np.array([1, 2, 3, 4, 5]), format="ratio")
            0.4714045207910317

            >>> cv_numpy(np.array([1, 2, 3, 4, 5]), format="percent")
            47.14045207910317
    """
    # Check if x is a numpy array
    if not isinstance(x, np.ndarray):
        try: 
            x = np.asarray(x)
        except:
            raise TypeError("Input x must be an array-like object.")
        
    # Check if axis is an integer
    if not isinstance(axis, int):
        raise TypeError("Input axis must be an integer. [0,1]")
    
    # Check if ddof is an integer
    if not isinstance(ddof, int):
        raise TypeError("Input ddof must be an integer.")

    # If ignore_nan use np.nanstd and np.nanmean
    if ignore_nan:
        cv = np.nanstd(x, axis=axis, ddof=ddof) / np.nanmean(x, axis=axis)
    else:
        cv = np.std(x, axis=axis, ddof=ddof) / np.mean(x, axis=axis)

    if format == "ratio":
        return cv
    elif format == "percent":
        return cv * 100

def scale_the_data(
        data: pd.DataFrame, 
        method: str="zscore",
        axis: int=1, 
        is_log: bool=False
    ):
    """
        Function to scale the data using different methods across the rows or columns.

        Args:
            data (pd.DataFrame): The data to be scaled.
            method (str, optional): The scaling method. Defaults to "zscore".
            axis (int, optional): The axis along which to scale the data. Defaults to 1.
            is_log (bool, optional): If the data is log-transformed. Defaults to False.

        Returns:
            pd.DataFrame: The scaled data.

        Raises:
            ValueError: If the data type is not supported.
            ValueError: If the axis is not 0 or 1.
            ValueError: If the method is not zscore, minmax, foldchange, log2, or log10.

        Examples:
    """

    # Check the data type
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "The data type is not supported, please use either a pandas DataFrame"
        )
    
    # Check the axis
    if axis not in [0, 1]:
        raise ValueError(
            "The axis should be either 0 (column-wise) or 1 (row-wise)."
        )

    # Check the method
    if method not in ["zscore", "minmax", "foldchange", "log2", "log10"]:
        raise ValueError(
            "The method should be either zscore, minmax, foldchange, log2, or log10."
        )

    idx = data.index
    cols = data.columns

    # Z-score Standardization
    if method == "zscore":
        if axis == 0:
            res = (
                data.values - 
                data.mean(axis=axis).values
            ) / data.std(axis=axis).values
        else:
            res = (
                data.values - 
                data.mean(axis=axis).values.reshape(-1, 1)
            ) / data.std(axis=axis).values.reshape(-1, 1)
    # Min-Max scaling
    elif method == "minmax":
        if axis == 0:
            res = (
                data.values - data.min(axis=axis).values
            ) / (
                data.max(axis=axis).values - 
                data.min(axis=axis).values
            )
        else:
            res = (
                data.values - 
                data.min(axis=axis).values.reshape(-1, 1)
            ) / (
                data.max(axis=axis).values.reshape(-1, 1) - 
                data.min(axis=axis).values.reshape(-1, 1)
            )
    # Fold-change scaling
    elif method == "foldchange":
        if axis == 0:
            if is_log:
                res = (
                    data.values - data.mean(axis=axis).values
                )
            else:
                res = (
                    data.values / data.mean(axis=axis).values
                )
        else:
            if is_log:
                res = (
                    data.values - 
                    data.mean(axis=axis).values.reshape(-1, 1)
                )
            else:
                res = (
                    data.values / 
                    data.mean(axis=axis).values.reshape(-1, 1)
                )
    elif method == "log2":
        res = np.log2(data.values)
    elif method == "log10":
        res = np.log10(data.values)

    
    return pd.DataFrame(
        res, 
        index=idx, 
        columns=cols
    )    

def knn_imputation(
        data: pd.DataFrame,
        n_neighbors: int = 5
    ) -> pd.DataFrame:
    """
        Imputes missing values in a DataFrame using the K-Nearest Neighbors algorithm.

        Args:
            data: The DataFrame with missing values (NaN).
            n_neighbors: Number of neighbors to use for imputation.

        Returns:
            A DataFrame with imputed values.

        Raises:
            ValueError: If n_neighbors is not a positive integer.
    """

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer.")

    df = data.copy()  
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

def iterative_imputation(data: pd.DataFrame, estimator=None) -> pd.DataFrame:
    """Imputes missing values using IterativeImputer with an optional estimator."""
    df = data.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer = IterativeImputer(estimator=estimator)  
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols]).round(2)
    return df

def linear_regression_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values using IterativeImputer with LinearRegression."""
    return iterative_imputation(data, estimator=LinearRegression())

def normalize_against_condition(
        long_data: pd.DataFrame,                   
        cond_run_dict: dict,                       
        run_col: str = "filename",                 
        index_cols: list = ["protein_id", "peptide_id"], 
        norm_against: str = "day1",                
        intensity_col: str = "intensity",          
        is_log2: bool = False,                     
        norm_intensity_col: str = "ms1adj"         

    ) -> pd.DataFrame:
    """
        Function to normalize the data against a condition for my method

        Args:
            long_data (pd.DataFrame): Long data with intensity values
            cond_run_dict (dict): Dictionary with condition: [run1, run2, ...]
            run_col (str): Column with the run names (used for pivoting)
            index_cols (list): Used for wide data and merging
            norm_against (str): Has to be one of the conditions
            intensity_col (str): Column with the intensity values
            is_log2 (bool): If the data is log2 transformed already
            norm_intensity_col (str): Column name for the normalized intensities

        Returns:
            pd.DataFrame: Long data with normalized intensities

        Raises:
            ValueError: If run_col, intensity_col, or index_cols are not in the long_data
            ValueError: If norm_against is not in the cond_run_dict keys

        Examples:
    """
    
    # index_cols must be a list
    if not isinstance(index_cols, list):
        index_cols = [index_cols]

    # check necessary columns if they are in the long_data
    for col in [run_col, intensity_col, *index_cols]:
        if col not in long_data.columns:
            raise ValueError(f"{col} not found in the columns of long_data")
        
    # Check if norm_against is key in the cond_run_dict with list of values
    if norm_against not in cond_run_dict.keys():
        raise ValueError(f"{norm_against} not found in the cond_run_dict keys")
    else:
        normCols = cond_run_dict[norm_against]
        
    wide_data = long_data.pivot_table(
        index=index_cols,
        columns=[run_col],
        values=intensity_col,
    )
    if not is_log2:
        wide_data = np.log2(wide_data)

    # Center the data by
    wide_data = (wide_data - wide_data.mean()) / wide_data.std()

    # Calculate the row-means for norm_against samples
    cntrRowMean = wide_data[normCols].mean(axis=1)
    # Subtract the row-means from the data
    wide_data = wide_data.sub(cntrRowMean, axis=0)

    # Return the data back to long format
    wide_data.reset_index().melt(
        id_vars=index_cols,
        var_name="filename",
        value_name=norm_intensity_col,
    )

    # Merge the normalized data with the original data
    long_data = pd.merge(
        # Original data
        long_data,
        # Return the data back to long format
        wide_data.reset_index().melt(
            id_vars=index_cols,
            var_name="filename",
            value_name=norm_intensity_col,
        ),
        # Merge on the index_cols and filename
        on=[*index_cols, "filename"],
        # Use left join
        how="left",
    )

    return long_data

def update_proteoform_grouping_in_COPF(
        data: pd.DataFrame,
        score_thr: float = 0.5,      # Specific to COPF if None, it will not be used
        pval_thr: float = 0.05,
        protein_col: str = "protein_id",
        cluster_col: str = "cluster",
        pval_col: str = "proteoform_score_pval_adj",
        score_col: str = "proteoform_score",
        sep: str = "_"
    ):
    # Create initial proteoform_id based on score and pval thresholds
    if score_thr is not None:
        condition = (data[score_col] >= score_thr) & (data[pval_col] <= pval_thr)
    else:
        condition = data[pval_col] <= pval_thr
    
    data['proteoform_id'] = data[protein_col]
    data.loc[condition, 'proteoform_id'] = data.loc[condition].apply(
        lambda row: f"{row[protein_col]}{sep}{int(row[cluster_col])}", axis=1
    )
    # If proteoform_id is exactly the same as protein_id, add "_0" to the cluster 0
    data.loc[data[protein_col] == data['proteoform_id'], 'proteoform_id'] = data[protein_col] + f"{sep}0"
    
    # Special case for cluster 100
    data.loc[data[cluster_col] == 100, 'proteoform_id'] = data[protein_col] + f"{sep}0"
    
    # Count unique proteoform_id per protein_id
    n_proteoforms = data.groupby(protein_col)['proteoform_id'].transform('nunique')
    
    # Adjust n_proteoforms if "_0" exists
    has_zero_cluster = data['proteoform_id'].str.endswith(f'{sep}0')
    n_proteoforms -= has_zero_cluster.groupby(data[protein_col]).transform('sum')
    
    # Final adjustment of proteoform_id
    data.loc[n_proteoforms.isin([0, 1]), 'proteoform_id'] = data[protein_col]
    
    return data

def generate_thresholds(
        base: float, 
        start_exp: int, 
        end_exp: int, 
        seq_start: float, 
        seq_end: float, 
        seq_step: float
    ) -> np.ndarray:
    """
        Generate a sorted array of unique values from the outer product 
        of a base raised to a range of exponents and a sequence of values.

        Args:
            base: The base for the outer product.
            start_exp: The start exponent for the outer product.
            end_exp: The end exponent for the outer product.
            seq_start: The start value of the sequence.
            seq_end: The end value of the sequence.
            seq_step: The step value of the sequence.

        Returns:
            A sorted array of unique values.
    """
    # Outer product with floating-point base
    outer_product = np.outer(1, base**np.arange(start_exp, end_exp + 1))

    # Sequence from seq_start to seq_end with step seq_step
    sequence = np.arange(seq_start, seq_end + seq_step, seq_step)

    # Combine arrays, get unique values, and sort
    result = np.sort(np.unique(np.concatenate((outer_product.flatten(), sequence))))

    # Don't keep values below 0 or above 1
    result = result[(result >= 0) & (result <= 1)]

    return result

def calculate_f1_score(
        precision: float,
        recall: float
    ) -> float:
    """
        Calculate the F1 Score using precision and recall values.

        Args:
            precision (float): The precision value.
            recall (float): The recall value.

        Returns:
            float: The F1 Score.
    """
    # Validations
    if (precision + recall) == 0: return 0

    # Calculate the F1 Score
    f1 = 2 * ((precision * recall) / (precision + recall))

    return f1

def build_proteoform_groups( # Specific for benchmark only
        cur_protein: str,
        data: pd.DataFrame,
        sample_col: str = "filename",
        peptide_col: str = "peptide_id",
        protein_col: str = "protein_id",
        quant_col: str = "log2_intensity", # Quant column to use for cluster    
        corrMethod: str = "kendall",
        distanceMethod: str = "euclidean",
        linkageMethod: str = "complete",
        minSizePct: float = 0.50,
        absoluteMaxSize: int = 3,
        aggfunc: str = "mean",
        verbose: bool = 0
    ):

    # Get the data for the current protein
    try: 
        subdata = data.loc[cur_protein]
    except :
        try: 
            subdata = data[data[protein_col] == cur_protein]
        except:
            raise ValueError("Protein not found in data")

    pep_idx = subdata[peptide_col].unique()

    # Convert long to wide format and calculate the correlation matrix
    corr_matrix  = subdata.pivot_table(
        index=peptide_col,
        columns=sample_col,
        values=quant_col,
        aggfunc=aggfunc
    ).loc[pep_idx].T.corr(method=corrMethod).values
    # Find the clusters
    clusters = _find_clusters_v2(
        corr_matrix, 
        distanceMethod, 
        linkageMethod, 
        minSizePct, 
        absoluteMaxSize,
        verbose = verbose
    )

    return pd.DataFrame({
        protein_col: cur_protein,
        peptide_col: pep_idx,
        "cluster_id": clusters
    })

def calculate_metrics(
        true_labels: pd.Series,
        pred_labels: pd.Series,        
        verbose: bool = False,
        return_metrics: bool = False
    ):
    """
        Calculate the True Positive Rate (TPR), False Positive Rate (FPR), 
        False Discovery Rate (FDR), Accuracy (ACC), Matthews Correlation Coefficient (MCC),
        and F1 Score based on a given threshold.

        Args:
            true_labels (pd.Series): The true labels.
            pred_labels (pd.Series): The predicted labels.
            verbose (bool): If True, print the metrics.
            return_metrics (bool): If True, return the metrics as a dictionary.

        Returns:
            dict: A dictionary containing the metrics.

    """
    
    # Calculate the confusion matrix
    conf_matrix = pd.crosstab(
        index=true_labels,      # True labels
        columns=pred_labels     # Predicted labels
    )
    # Reindex to ensure both True and False are present for both columns and rows
    conf_matrix = conf_matrix.reindex(index=[True, False], columns=[True, False], fill_value=0)
    
    # Extract the values from the confusion matrix
    TP = conf_matrix.loc[True, True]
    FP = conf_matrix.loc[False, True]
    TN = conf_matrix.loc[False, False]
    FN = conf_matrix.loc[True, False]   

    # Calculate metrics
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # Handle potential division by zero
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FDR = FP / (FP + TP) if (FP + TP) != 0 else 0
    # MCC = calculate_mcc(TP, TN, FP, FN)
    MCC = matthews_corrcoef(true_labels, pred_labels)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = calculate_f1_score(precision, recall)

    if verbose:
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Metrics:\n==========")
        for metric, value in zip(
            ["TPR", "FPR", "FDR", "MCC", "F1", "Precision", "Recall"],
            [TPR, FPR, FDR, MCC, F1, precision, recall]
        ):
            print(f" {metric}: {value:.2f}")

    if return_metrics:
        return {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            
            "TPR": TPR,
            "FPR": FPR,
            "FDR": FDR,
            "MCC": MCC,

            "Precision": precision,
            "Recall": recall,
            "F1": F1
        }

def create_metric_data(
        data: pd.DataFrame, 
        pvalue_thresholds: list,
        label_col: str ="perturbed_peptide", 
        pvalue_col: str ="adj.pval", 
    ):
    """
        Create a DataFrame containing performance metrics for various p-value and score thresholds.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            pvalue_thresholds (iterable): The p-value thresholds to evaluate.
            label_col (str): The column with the true labels.
            pvalue_col (str): The column with the threshold values (p-values).

        Returns:
            pd.DataFrame: A DataFrame containing the metrics for each threshold combination.
    """

    # Initialize a list to store metric dictionaries
    metrics_data = []

    # Iterate over p-value and score thresholds
    for pthr in pvalue_thresholds:
        true_labels = data[label_col]
        pred_labels = data[pvalue_col] <= pthr
        # Calculate metrics for the current threshold combination
        metrics = calculate_metrics(
            true_labels=true_labels, pred_labels=pred_labels, 
            verbose=False, return_metrics=True
        )

        # Add threshold values to the metrics dictionary
        metrics["threshold"] = pthr

        # Append the metrics dictionary to the list
        metrics_data.append(metrics)

    # Create a DataFrame from the list of metric dictionaries
    return pd.DataFrame(metrics_data)

########################## Simulation Functions ##########################

def sample_peptides(
        peptides: pd.Series,
        n: int
    ) -> pd.Series:
    """
        Utility function to sample n peptides from the given
        peptides Series

        Args:
            peptides (pd.Series): Peptides Series
            n (int): Number of peptides to sample
        
        Returns:
            pd.Series: Series with sampled peptides
    """
    if n > 0 and n <= len(peptides):
        return peptides.isin(np.random.choice(peptides, n, replace=False))
    else:
        return peptides.isin([])

def calculate_nPerturb(
        nPeptides: int,
        perturbNumber: str
    ) -> int:
    """
        Utility function to calculate the number of 
        peptides to perturb based on the perturbNumber

        Args:
            nPeptides (int): Number of peptides
            perturbNumber (str): Perturbation number
        
        Returns:
            int: Number of peptides to perturb
    """
    if perturbNumber == "one":
        nPerturb = 1
    elif perturbNumber == "two":
        nPerturb = 2
    elif perturbNumber == "quarter":
        nPerturb = nPeptides // 4
    elif perturbNumber == "half":
        nPerturb = nPeptides // 2
    elif perturbNumber == "random":
        nPerturb = np.random.randint(2, nPeptides // 2 + 1)
    return nPerturb

def perturb_proteins(
        data: pd.DataFrame,                 # Long format data with 
        info: pd.DataFrame,                 # Info data with protein-peptide info
        proteinPertDict: dict,              # Dictionary with protein:red_factor (perturbed proteins)
        perturbN: str = "random",           # Number of peptides to perturb (one, two, quarter, half, random)
        proteinCol: str = "Protein",        # Protein column name
        peptideCol: str = "Mod.Peptide",    # Peptide column name
        intensityCol: str = "Intensity",    # Intensity column name
        conditionCol: str = "Condition",    # Condition column name
        condition2Perturb: str = "day5"     # Condition to perturb

    ) -> pd.DataFrame:

    """
        Function to perturb the proteins in the data based on the
        proteinPertDict and perturbN settings

        Args:
            data (pd.DataFrame): Long format data with protein-peptide info
            info (pd.DataFrame): Info data with protein-peptide info
            proteinPertDict (dict): Dictionary with protein:red_factor (perturbed proteins)
            perturbN (str): Number of peptides to perturb (one, two, quarter, half, random)
            proteinCol (str): Protein column name
            peptideCol (str): Peptide column name
            intensityCol (str): Intensity column name
            conditionCol (str): Condition column name
            condition2Perturb (str): Condition to perturb

        Returns:
            pd.DataFrame: Data with perturbed proteins
    """
    
    # Get the unique proteins set to perturb
    perturbed_proteins = list(proteinPertDict.keys())
    # Store the perturbed protein information (boolean)
    info["perturbed_protein"] = info[proteinCol].isin(perturbed_proteins)
    # Store the number of peptides per protein
    info["nPeps"] = info[proteinCol].map(info.groupby(proteinCol).size())
    # Store the number of peptides to perturb based on the perturbN setting
    info["nPerPeps"] = info[proteinCol].map(
        info.groupby(proteinCol).size().map(
            lambda x: calculate_nPerturb(x, perturbN)
        )
    )
    # If perturb_protein is False, set nPerPeps to 0
    info.loc[~info["perturbed_protein"], "nPerPeps"] = 0
    # Store the red factor for the perturbed proteins
    info["red_factor"] = info[proteinCol].map(proteinPertDict).fillna(1)
    # Randomly pick nPerPeps peptides from each protein
    info["perturbed_peptide"] = info.groupby(proteinCol)[peptideCol].transform(
        lambda x: sample_peptides(x, info.loc[x.index, "nPerPeps"].iloc[0])
    )
    
    # Merge the info data with the data
    data = data.merge( info, on=[proteinCol, peptideCol], how="left" )

    # If the peptide is perturbed and the condition is the one to perturb, 
    #  multiply the intensity by the red factor
    # else keep the intensity as it is
    # 
    data["PerturbedIntensity"] = np.where(
        (data["perturbed_peptide"]) & (data[conditionCol] == condition2Perturb),
        data[intensityCol] * data["red_factor"],
        data[intensityCol]
    )   

    return data


########################## Correlation-Based Proteoform Groups ##########################

def calculate_quality_score(
        mean: float,
        median: float,
        std: float,
        var: float,
        size: int,
        method: str = 'weighted_sum_with_cv'
    ) -> float:
    """
        Calculate a quality score for a given set of statistics.

        Args:
            mean (float): Mean value.
            median (float): Median value.
            std (float): Standard deviation.
            var (float): Variance.
            size (int): Size of the data.
            method (str): Method to calculate the quality score.

        Returns:
            float: The quality score.

        Raises:
            ValueError: If the method is not recognized.
            ZeroDivisionError: If the mean is zero when calculating CV.
    """

    if method == 'weighted_sum_with_cv':
        # Define weights for each statistic
        weights = {
            'mean': 0.4,
            'median_deviation': 0.15,  # Penalize deviation from median
            'cv': 0.3,
            'size': 0.15
        }

        if mean == 0:
            raise ZeroDivisionError("Mean cannot be zero when calculating Coefficient of Variation.")

        cv = std / mean  # Calculate Coefficient of Variation

        # Calculate weighted sum
        score = (
            weights['mean'] * mean -
            weights['median_deviation'] * abs(mean - median) -
            weights['cv'] * cv +
            weights['size'] * size
        )
    elif method == 'weighted_sum':
        # Define weights for each statistic
        weights = {
            'mean': 0.4,
            'median': 0.15,
            'std': 0.3,
            'size': 0.15
        }

        # Calculate weighted sum
        score = (
            weights['mean'] * mean -
            weights['std'] * std +
            weights['size'] * size
        )
    elif method == 'custom':
        # Define a custom formula for the quality score
        score = mean / (var + 1) * size

    elif method == 'z_scored':
        # Calculate the z-score of the mean
        z_score = (mean - median) / (std+1e-6)
        # Calculate the quality score
        score = z_score * size

    else:
        raise ValueError("Unknown method: {}".format(method))
    
    return score

# TODO: Better Name
def make_res_dict(
        distance,
        cluster,
        cluster_indices,
        corr_matrix,
        clusterScoreMethod='custom'
    ):
    """
        TODO: Make a better description
    """
    # Subset the corr_matrix with the cluster indices
    cluster_corr = corr_matrix[cluster_indices][:, cluster_indices]
    m = np.nanmean(cluster_corr)
    s = np.nanstd(cluster_corr)
    v = np.nanvar(cluster_corr)
    md = np.nanmedian(cluster_corr)
    n = len(cluster_indices)
    
    res = {
        "Distance": distance,
        "Cluster": cluster,
        "Members": cluster_indices.tolist(),
        "Mean": m,
        "Var": v,
        "Size": n,
        "Score": calculate_quality_score(m, md, s, v, n, method=clusterScoreMethod),
    }

    return res

def get_significant_ids(tmp):
    signArr = tmp["isSignificant"].values
    pepIDArr = tmp["PeptideID"].values

    return pepIDArr[signArr]

def find_clusters_with_significant_member(unique_members, significant_ids, mapper=None):
    # Added a mapper to go from the text indices to numeric indices if needed
    if mapper is not None:
        # Modify the significant_ids based on the mapper
        significant_ids = [mapper[i] for i in significant_ids]
        
    member_map = {}
    for i in unique_members:
        l = eval(i)
        # Check if the member contains any value in b
        if any(x in significant_ids for x in l):
            member_map[i] = True
        else:
            member_map[i] = False
    
    return member_map

def select_best_distance(data, method="sum"):
    # if method not in ["max", "mean", "min", "sum", "median", "std"]:
    #     raise ValueError("Invalid method provided, please choose from max, mean, min, sum, median, std")

    scores = data.groupby("Distance")["Combined_Score"].agg(method).reset_index()

    # Sort the distances by score (descending) and distance (ascending)
    scores = scores.sort_values(by=["Combined_Score", "Distance"], ascending=[False, True])
    
    # Iterate through the sorted distances and check for the first distance with a significant cluster
    for distance in scores["Distance"]:
        if data.loc[data["Distance"] == distance, "Significant"].sum() > 0:
            return data.loc[data["Distance"] == distance]
    
    # If no significant cluster is found, return None
    return None

def _find_clusters(
        corr_matrix: np.ndarray,
        distanceMethod: str = "euclidean",
        linkageMethod: str = "complete",
        minSizePct: int = 0.25,
        verbose: bool = 0
    ) -> np.ndarray:
        """
            Utility function to find clusters based on the correlation matrix, using the
                specified distance and linkage methods with minSizePct passed to dynamicTreeCut.

            Args:
                corr_matrix (np.ndarray): The correlation matrix.
                distanceMethod (str): The distance method to use.
                linkageMethod (str): The linkage method to use.
                minSizePct (int): The minimum size percentage for the clusters.
                verbose (bool): Whether to print additional information.
            
            Returns:
                np.ndarray: The cluster labels.
        """
        
        nPeptides = len(corr_matrix)
        minClusterSize = max(1, int(nPeptides * minSizePct)) # Minimum cluster size
        if verbose: print(" - minClusterSize =", minClusterSize)
        # Calculate the distance matrix
        distance_matrix = 1 - corr_matrix
        # Calculate the distances
        distances = pdist(distance_matrix, metric=distanceMethod)
        # Calculate the linkage matrix
        link = linkage(distances, method=linkageMethod)
        # Calculate the clusters
        clusters = cutreeHybrid(link, distances, minClusterSize=minClusterSize, verbose=verbose)
        return clusters['labels']



def _find_clusters_v2(
        corr_matrix: np.ndarray,
        distanceMethod: str = "euclidean",
        linkageMethod: str = "complete",
        maxSizePct: float = 0.75,
        absoluteMaxSize: int = 3,
        verbose: bool = 0
    ) -> np.ndarray:
    """
    Utility function to find clusters based on the correlation matrix, using the
    specified distance and linkage methods with maxSizePct passed to cut_balanced.

    Args:
        corr_matrix (np.ndarray): The correlation matrix.
        distanceMethod (str): The distance method to use.
        linkageMethod (str): The linkage method to use.
        maxSizePct (float): The maximum size percentage for the clusters.
        verbose (bool): Whether to print additional information.

    Returns:
        np.ndarray: The cluster labels.
    """
    
    nPeptides = len(corr_matrix)
    maxClusterSize = max(absoluteMaxSize, int(nPeptides * maxSizePct))  # Maximum cluster size
    if verbose: print(" - maxClusterSize =", maxClusterSize)
    
    # Calculate the distance matrix
    distance_matrix = 1 - corr_matrix
    
    # Calculate the distances
    distances = pdist(distance_matrix, metric=distanceMethod)
    try:
        # Calculate the linkage matrix
        link = linkage(distances, method=linkageMethod)
    except:
        # Ensure the correlation matrix contains only finite values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        # Calculate the distance matrix
        distance_matrix = 1 - corr_matrix
        # Calculate the distances
        distances = pdist(distance_matrix, metric=distanceMethod)
        # Calculate the linkage matrix
        link = linkage(distances, method=linkageMethod)
        
    # Calculate the clusters using cut_balanced
    clusters = cut_balanced(link, max_cluster_size=maxClusterSize)
    
    return clusters

def peptide_clusters(
        cur_protein: str,
        quant_data: pd.DataFrame,
        info_data: pd.DataFrame,
        data_format: str = "wide", # wide (only quant) or long (quant columns + info columns)
        quantcol: str = "ms1adj",
        samplecol: str = "filename",  

        corrMethod: str = "spearman",
        distanceMethod: str = "euclidean",
        linkageMethod: str = "complete",
        minSizePct: float = 0.25,
        aggfunc: str = "mean",

        # Column Definitions
        proteinCol: str = "Protein",
        peptideCol: str = "Peptide",
        peptideIDCol: str = "PeptideID",

        verbose: bool = 0
    ):

    """

    """
    # TODO: This is still problematic with multiple usecases, need to standardize

    if verbose: print("Analyzing Protein:", cur_protein)
    # Subset for protein
    subset_info = info_data.loc[cur_protein].set_index([peptideCol])
    subset_quan = quant_data.loc[cur_protein]
    subset_quan[peptideIDCol] = subset_quan[peptideCol].map(subset_info[peptideIDCol].to_dict())
    
    # If data is long create wide data
    if data_format == "long":
        # Check if required columns are present
        if not set([samplecol, quantcol]).issubset(subset_quan.columns):
            raise ValueError("Required columns not found in the long data for conversion to wide.")
        wide_data = subset_quan.pivot_table(
            index=peptideIDCol,
            columns=samplecol,
            values=quantcol,
            aggfunc=aggfunc
        )
    elif data_format == "wide":
        wide_data = subset_quan.set_index(peptideIDCol)
        # Check if any columns are non-numeric
        if subset_quan.select_dtypes(exclude=[np.number]).shape[1] > 0:
            raise ValueError("Non-numeric columns found in the wide data.")
    else:
        raise ValueError("Invalid data_format provided, please choose from wide or long")
        
    # Create a correlation matrix for the protein
    corr_matrix = wide_data.T.corr(method=corrMethod).values 
    # Find the clusters
    clusters = _find_clusters(corr_matrix, distanceMethod, linkageMethod, minSizePct)
    
    # Check if the clusters are valid (labels)
    try:
        return pd.DataFrame({
            proteinCol: cur_protein,
            "PeptideID": subset_info[peptideIDCol],
            "Cluster": clusters
        }).reset_index().set_index([proteinCol, peptideCol])
    except :
        return pd.DataFrame({
            proteinCol: cur_protein,
            "PeptideID": subset_info[peptideIDCol],
            "Cluster": np.zeros(len(subset_info))-1
        }).reset_index().set_index([proteinCol, peptideCol])

from collections import defaultdict

def expand_mismatches(x, y, reverse_mapping=False):
  """
    Expands mismatches between two dictionaries with clusterID: [PeptideIDs] 
        mappings, handling cases where clusters might be missing in one 
        of the inputs.

    Args:
        x: The first dictionary.
        y: The second dictionary.
        reverse_mapping: If True, returns a dictionary with PeptideID: clusterID mapping.
                        If False (default), returns a dictionary with clusterID: [PeptideIDs] mapping.

    Returns:
        A dictionary with the specified mapping.
  """

  mapping = defaultdict(list)
  cluster_id = 0

  x_keys = set(x.keys())
  y_keys = set(y.keys())
  all_keys = x_keys | y_keys

  for key in all_keys:
    x_values = set(x.get(key, []))
    y_values = set(y.get(key, []))

    matches = x_values & y_values
    for match in matches:
      mapping[cluster_id].append(match)
    cluster_id += 1

    x_mismatches = x_values - y_values
    for mismatch in x_mismatches:
      mapping[cluster_id].append(mismatch)
    cluster_id += 1

    y_mismatches = y_values - x_values
    for mismatch in y_mismatches:
      mapping[cluster_id].append(mismatch)
    cluster_id += 1

  if reverse_mapping:
    reversed_mapping = {}
    for cluster_id, peptide_ids in mapping.items():
      for peptide_id in peptide_ids:
        reversed_mapping[peptide_id] = cluster_id
    return reversed_mapping
  else:
    return mapping
  
def process_data(data):
    """
        Processes the data to determine dPFs based on two clusterings and significance.

        Args:
            data: A pandas DataFrame with columns "Protein", "cluster_id", 
                "oldCluster", "PeptideID", and "isSignificant".

        Returns:
            The DataFrame with added "newClusters" and "new_dPFs" columns.
    """

    def apply_expand_mismatches(group):
        x = group.groupby("cluster_id")["PeptideID"].apply(list).to_dict()
        y = group.groupby("oldCluster")["PeptideID"].apply(list).to_dict()
        consensus = expand_mismatches(x, y, reverse_mapping=True)
        group["newClusters"] = group["PeptideID"].map(consensus)
        return group

    def calculate_dpfs(group):
        dpf_id = 0
        sig_clusters = group.groupby("newClusters")["isSignificant"].any()  # Clusters with at least one significant peptide
        for cluster_id, has_sig_peptide in sig_clusters.items():
            cluster_peptides = group[group["newClusters"] == cluster_id]
            if not has_sig_peptide:
                group.loc[cluster_peptides.index, "new_dPFs"] = 0  # No significant peptides
            elif len(cluster_peptides) == 1:
                group.loc[cluster_peptides.index, "new_dPFs"] = -1  # Single significant peptide
            else:
                dpf_id += 1
                group.loc[cluster_peptides.index, "new_dPFs"] = dpf_id  # Multiple peptides with at least one significant

        return group

    data = data.groupby("Protein").apply(apply_expand_mismatches).reset_index(drop=True)
    data = data.groupby("Protein").apply(calculate_dpfs).reset_index(drop=True)
    return data