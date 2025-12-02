#!/usr/bin/env python3
"""
UniProt FASTA Parser and Protein Analysis Utilities

A high-performance, lightweight parser for UniProt FASTA files with utilities
for calculating protein molecular weights and other properties. Optimized for
minimal memory usage and maximum speed when processing large proteome files.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import re
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from typing import (
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd


# ======================================================================================
# Constants and Configuration
# ======================================================================================


# Monoisotopic residue masses of the standard amino acids.
# Source: ExPASy ProtParam tool (widely accepted values)
AMINO_ACID_WEIGHTS = {
    'A': 71.03711,   'C': 103.00919,  'D': 115.02694,  'E': 129.04259,
    'F': 147.06841,  'G': 57.02146,   'H': 137.05891,  'I': 113.08406,
    'K': 128.09496,  'L': 113.08406,  'M': 131.04049,  'N': 114.04293,
    'P': 97.05276,   'Q': 128.05858,  'R': 156.10111,  'S': 87.03203,
    'T': 101.04768,  'V': 99.06841,   'W': 186.07931,  'Y': 163.06333,
}

# Extended amino acids
EXTENDED_AMINO_ACIDS = {
    'U': 150.948923,  # Selenocysteine
    'O': 237.143012,  # Pyrrolysine
}

# Merge extended amino acids into the main weights dictionary
AMINO_ACID_WEIGHTS.update(EXTENDED_AMINO_ACIDS)

# Generate from the dictionary to ensure consistency
VALID_AMINO_ACIDS = set(AMINO_ACID_WEIGHTS.keys())
VALID_AMINO_ACIDS_STRING = "".join(sorted(AMINO_ACID_WEIGHTS.keys()))

# Molecular weight of water (H2O) - added to account for peptide bonds
WATER_WEIGHT = 18.01528

# Default column order for DataFrame output
DEFAULT_COLUMNS = [
    "entry", "entryName", "geneName", "proteinDescription",
    "reviewStatus", "isoformStatus", "sequenceLength", 
    "molecularWeight_kDa", "sequence"
]

# Regular expressions for header parsing (pre-compiled for performance)
HEADER_PATTERNS = {
    'organism': re.compile(r'OS=([^=]+?)(?:\s+(?:OX|GN|PE|SV)=|$)'),
    'gene': re.compile(r'GN=([^\s]+)'),
    'existence': re.compile(r'PE=(\d+)'),
    'version': re.compile(r'SV=(\d+)'),
    'taxonomy': re.compile(r'OX=(\d+)'),
}

# Pre-compiled regex for sequence validation (much faster than set operations)
INVALID_AA_PATTERN = re.compile(f'[^{VALID_AMINO_ACIDS_STRING}]')
NON_STANDARD_CHARS_PATTERN = re.compile(r'[^A-Z\s]')


# ======================================================================================
# Performance and Multiprocessing Utilities
# ======================================================================================


def _estimate_file_size(
        file_path: Union[str, Path]
    ) -> int:
    """
    Estimate the number of entries in a FASTA file by sampling.
    
    Args:
        file_path: Path to the FASTA file
        
    Returns:
        Estimated number of entries
    """
    path = Path(file_path)
    file_size = path.stat().st_size
    
    # Sample first 1MB or entire file if smaller
    sample_size = min(1024 * 1024, file_size)
    
    with open(path, 'r', encoding='utf-8') as f:
        sample = f.read(sample_size)
    
    # Count headers in sample
    header_count = sample.count('\n>')
    if sample.startswith('>'):
        header_count += 1
    
    # Estimate total entries
    if sample_size == file_size:
        return header_count
    else:
        return int((header_count * file_size) / sample_size)


def _get_optimal_processing_params(
        file_path: Union[str, Path]
    ) -> Tuple[bool, int, int]:
    """
    Automatically determine optimal processing parameters based on file size.
    
    Conservative approach that only uses multiprocessing when the benefit 
    significantly outweighs the overhead.
    
    Args:
        file_path: Path to the FASTA file
        
    Returns:
        Tuple of (use_multiprocessing, n_processes, chunk_size)
    """
    estimated_entries = _estimate_file_size(file_path)
    
    # Much more conservative thresholds based on real-world performance
    if estimated_entries <= 5000:
        # Small to medium files: single process is usually faster
        return False, 1, 0
    elif estimated_entries <= 50000:
        # Large files: use minimal multiprocessing
        # Only use 2-4 processes to minimize overhead
        n_processes = min(4, max(2, cpu_count() // 4))
        chunk_size = max(1000, estimated_entries // (n_processes * 2))
        return True, n_processes, chunk_size
    else:
        # Very large files: more aggressive but still conservative
        # Cap at 8 processes - more usually hurts performance
        n_processes = min(8, max(2, cpu_count() // 2))
        chunk_size = max(2000, estimated_entries // (n_processes * 3))
        return True, n_processes, chunk_size


def _chunk_fasta_entries(
        entries: List[Tuple[str, str]], 
        chunk_size: int
    ) -> List[List[Tuple[str, str]]]:
    """
    Split FASTA entries into chunks for parallel processing.
    
    Args:
        entries: List of (header, sequence) tuples
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks, each containing a list of (header, sequence) tuples
    """
    return [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]


def _process_chunk(
        args: Tuple[List[Tuple[str, str]], int, int, bool, bool]
    ) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Process a chunk of FASTA entries in parallel.
    
    Args:
        args: Tuple containing (chunk, min_length, max_length, validate_sequences, gene_only)
        
    Returns:
        Tuple of (processed_entries, skipped_counts)
    """
    chunk, min_length, max_length, validate_sequences, gene_only = args
    
    processed_entries = []
    skipped_counts = {
        'invalid_sequence': 0,
        'length_filter': 0,
        'gene_filter': 0,
        'molecular_weight_error': 0
    }
    
    for header, sequence in chunk:
        try:
            entry_data, skip_reason = _process_single_entry(
                header, sequence, min_length, max_length, 
                validate_sequences, gene_only, verbose=False  # Disable verbose in workers
            )
            
            if entry_data:
                processed_entries.append(entry_data)
            elif skip_reason:
                skipped_counts[skip_reason] += 1
                
        except Exception:
            # Skip problematic entries silently in worker processes
            continue
    
    return processed_entries, skipped_counts


# ======================================================================================
# FASTA Parsing and Validation Functions
# ======================================================================================

def read_fasta_entries(
        file_path: Union[str, Path], 
        verbose: bool = True
    ) -> Generator[Tuple[str, str], None, None]:
    """
    Fast generator that yields FASTA entries as (header, sequence) tuples.
    
    Args:
        file_path: Path to the FASTA file
        verbose: If True, print warnings and info messages
        
    Yields:
        Tuple[str, str]: A tuple containing (header, sequence) where header
                       does not include the leading '>' character
    """
    # Validate file path
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as file:
            header = None
            sequence_lines = []
            line_number = 0
            
            for line in file:
                line_number += 1
                line = line.rstrip('\n\r')  # Faster than strip() for just newlines
                
                if not line:  # Skip empty lines
                    continue
                    
                if line[0] == '>':  # Faster than startswith for single character
                    # Yield previous entry if exists
                    if header is not None:
                        sequence = "".join(sequence_lines)
                        if not sequence and verbose:
                            print(f"Warning: Empty sequence for header: {header}")
                        yield header, sequence
                    
                    # Start new entry
                    header = line[1:]  # Remove '>'
                    sequence_lines = []
                    
                elif header is not None:
                    # Add sequence line (convert to uppercase)
                    upper_line = line.upper()
                    if verbose and NON_STANDARD_CHARS_PATTERN.search(upper_line):
                        print(f"Warning: Non-standard characters in sequence at line {line_number}: {line[:50]}...")
                    sequence_lines.append(upper_line)
                    
                else:
                    raise ValueError(f"Sequence data found before header at line {line_number}")
            
            # Yield the last entry
            if header is not None:
                sequence = "".join(sequence_lines)
                yield header, sequence
            elif line_number == 0:
                raise ValueError("Empty FASTA file")
                
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error: {e}")
    except Exception as e:
        if not isinstance(e, (ValueError, FileNotFoundError)):
            raise ValueError(f"Error reading FASTA file: {e}")
        raise


def read_all_fasta_entries(
        file_path: Union[str, Path], 
        verbose: bool = True
    ) -> List[Tuple[str, str]]:
    """
    Read all FASTA entries into memory for multiprocessing.
    
    Args:
        file_path: Path to the FASTA file
        verbose: If True, print progress information
        
    Returns:
        List of (header, sequence) tuples
    """
    entries = []
    for header, sequence in read_fasta_entries(file_path, verbose=verbose):
        entries.append((header, sequence))
    return entries


def is_valid_sequence(
        sequence: str, 
        verbose: bool = True
    ) -> bool:
    """
    Check if a protein sequence contains only valid amino acid characters.
    
    Args:
        sequence: Protein sequence to validate
        verbose: If True, print warnings for invalid sequences
        
    Returns:
        bool: True if sequence contains only valid amino acids, False otherwise
    """
    if not sequence:
        if verbose:
            print("Warning: Empty sequence provided")
        return False

    # Pre-compiled regex is much faster than set operations
    upper_seq = sequence.upper()
    invalid_match = INVALID_AA_PATTERN.search(upper_seq)
    
    if invalid_match and verbose:
        print(f"Warning: Invalid characters in sequence: {sequence[:15]}...")
        
    return invalid_match is None


def calculate_molecular_weight(
        sequence: str, 
        in_kda: bool = False, 
        verbose: bool = True
    ) -> float:
    """
    Calculate the monoisotopic molecular weight of a protein sequence.
    
    Args:
        sequence: Amino acid sequence (e.g., "PEPTIDE")
        in_kda: If True, return weight in kDa; otherwise in Da
        verbose: If True, print warnings for invalid sequences
        
    Returns:
        float: Molecular weight in Da (default) or kDa
    """
    if not sequence:
        raise ValueError("Empty sequence provided")
        
    # Direct amino acid counting is faster than Counter for typical protein sequences
    upper_seq = sequence.upper()
    weight = WATER_WEIGHT
    
    # Fast counting using string.count() method
    for aa, aa_weight in AMINO_ACID_WEIGHTS.items():
        count = upper_seq.count(aa)
        if count > 0:
            weight += count * aa_weight
    
    # Quick validation - check if all characters were accounted for
    total_counted = sum(upper_seq.count(aa) for aa in AMINO_ACID_WEIGHTS)
    if total_counted != len(upper_seq):
        # Find the invalid character for error reporting
        invalid_chars = set(upper_seq) - VALID_AMINO_ACIDS
        if invalid_chars:
            invalid_char = next(iter(invalid_chars))
            raise ValueError(
                f"Invalid amino acid '{invalid_char}' found in sequence. "
                f"Valid amino acids: {VALID_AMINO_ACIDS_STRING}"
            )
    
    return weight / 1000.0 if in_kda else weight


def parse_uniprot_header(
        header: str, 
        verbose: bool = True
    ) -> Dict[str, Optional[str]]:
    """
    Parse a UniProt FASTA header into structured data.
    
    Args:
        header: UniProt FASTA header string (without leading '>')
        verbose: If True, print warnings for malformed headers
        
    Returns:
        Dict containing parsed fields
    """
    result = {
        'reviewStatus': None,
        'entry': None,
        'entryName': None,
        'proteinDescription': None,
        'geneName': None,
        'organism': None,
        'taxonomyId': None,
        'proteinExistence': None,
        'sequenceVersion': None,
    }
    
    # Split by pipe to get main components
    parts = header.split('|')
    if len(parts) < 3:
        if verbose:
            print(f"Warning: Malformed header: {header}")
        return result
        
    # Parse database and review status
    db_code = parts[0].strip()
    result['reviewStatus'] = 'reviewed' if db_code == 'sp' else 'unreviewed'
    
    # Parse accession
    result['entry'] = parts[1].strip()
    
    # Parse entry name and description
    name_desc_part = parts[2]
    name_desc_split = name_desc_part.split(' ', 1)
    result['entryName'] = name_desc_split[0]
    
    if len(name_desc_split) > 1:
        description_part = name_desc_split[1]
        
        # Extract protein description (everything before OS=)
        desc_match = description_part.split(' OS=')[0]
        result['proteinDescription'] = desc_match.strip()
        
        # Extract other fields using regex patterns
        for field, pattern in HEADER_PATTERNS.items():
            match = pattern.search(description_part)
            if match:
                value = match.group(1).strip()
                if field == 'organism':
                    result['organism'] = value
                elif field == 'gene':
                    result['geneName'] = value
                elif field == 'existence':
                    result['proteinExistence'] = int(value)
                elif field == 'version':
                    result['sequenceVersion'] = int(value)
                elif field == 'taxonomy':
                    result['taxonomyId'] = int(value)
    
    return result


def _process_single_entry(
        header: str, 
        sequence: str, 
        min_length: int, 
        max_length: int, 
        validate_sequences: bool, 
        gene_only: bool, 
        verbose: bool = False
    ) -> Tuple[Optional[Dict[str, any]], Optional[str]]:
    """
    Optimized single FASTA entry processor with minimal function call overhead.
    """
    # 1. Quick length check first (fastest filter)
    seq_length = len(sequence)
    if seq_length < min_length or seq_length > max_length:
        return None, 'length_filter'

    # 2. Validate sequence if requested (inline for performance)
    if validate_sequences:
        upper_sequence = sequence.upper()
        if INVALID_AA_PATTERN.search(upper_sequence):
            if verbose:
                print(f"Skipping sequence with invalid characters: {header[:50]}...")
            return None, 'invalid_sequence'
    else:
        upper_sequence = sequence.upper()

    # 3. Parse header efficiently
    parsed_header = parse_uniprot_header(header, verbose=False)

    # 4. Apply gene name filter early
    gene_name = parsed_header.get('geneName')
    if gene_only and not gene_name:
        return None, 'gene_filter'

    # 5. Calculate molecular weight with optimized method
    try:
        # Inline molecular weight calculation for performance
        weight = WATER_WEIGHT
        for aa, aa_weight in AMINO_ACID_WEIGHTS.items():
            count = upper_sequence.count(aa)
            if count > 0:
                weight += count * aa_weight
        
        # Quick validation
        total_counted = sum(upper_sequence.count(aa) for aa in AMINO_ACID_WEIGHTS)
        if total_counted != seq_length:
            raise ValueError("Invalid amino acid found")
            
        mol_weight_kda = weight / 1000.0
        
    except ValueError:
        if verbose:
            print(f"Could not calculate molecular weight for: {header[:50]}...")
        return None, 'molecular_weight_error'

    # 6. Assemble entry data (minimize dictionary operations)
    entry = parsed_header.get('entry', '')
    
    return {
        'entry': entry,
        'entryName': parsed_header.get('entryName'),
        'geneName': gene_name,
        'proteinDescription': parsed_header.get('proteinDescription'),
        'reviewStatus': parsed_header.get('reviewStatus'),
        'organism': parsed_header.get('organism'),
        'taxonomyId': parsed_header.get('taxonomyId'),
        'proteinExistence': parsed_header.get('proteinExistence'),
        'sequenceVersion': parsed_header.get('sequenceVersion'),
        'sequenceLength': seq_length,
        'molecularWeight_kDa': mol_weight_kda,
        'sequence': sequence,
        'isoformStatus': 'isoform' if '-' in entry else 'canonical'
    }, None


# ======================================================================================
# Main Processing Function
# ======================================================================================

def fasta_to_df(
        fasta_path: Union[str, Path],
        gene_only: bool = False,
        min_length: int = 7,
        max_length: int = 10**6,
        column_order: Optional[List[str]] = None,
        sort_by: Optional[List[str]] = None,
        sort_ascending: Optional[List[bool]] = None,
        validate_sequences: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
    """
    Process a UniProt FASTA file into a structured pandas DataFrame.
    
    High-performance implementation with automatic optimization for optimal
    processing method (single-process vs multiprocess) based on file size.
    All processing parameters are automatically determined for best performance.
    
    Args:
        fasta_path: Path to the UniProt FASTA file
        gene_only: If True, only include sequences with gene names
        min_length: Minimum sequence length to include (default: 7 AA)
        max_length: Maximum sequence length to include (default: 1M AA)
        column_order: Custom column order for output DataFrame
        sort_by: List of columns to sort by (default: ["entry", "isoformStatus"])
        sort_ascending: Sort order for each sort column
        validate_sequences: Whether to validate amino acid sequences
        verbose: If True, print information about processing; if False, be silent
        
    Returns:
        pd.DataFrame: Processed protein data
    """
    # Set defaults
    if column_order is None:
        column_order = DEFAULT_COLUMNS.copy()
    if sort_by is None:
        sort_by = ["entry", "isoformStatus"]
    if sort_ascending is None:
        sort_ascending = [True, False]
    
    # Automatically determine optimal processing parameters
    use_multiprocessing, n_processes, chunk_size = _get_optimal_processing_params(fasta_path)
    
    if verbose:
        print(f"Processing FASTA file: {fasta_path}")
        if use_multiprocessing:
            print(f"Auto-selected: multiprocessing with {n_processes} processes")
        else:
            print("Auto-selected: single-process mode (optimal for small files)")
    
    # Process based on auto-determined method
    if use_multiprocessing:
        # Multiprocessing path with performance monitoring
        if verbose:
            print("Loading FASTA entries...")
        
        import time
        load_start = time.time()
        all_entries = read_all_fasta_entries(fasta_path, verbose=False)
        load_time = time.time() - load_start
        
        if not all_entries:
            if verbose:
                print("Warning: No entries found in the file")
            return pd.DataFrame(columns=column_order)
        
        # Quick performance check: if loading took too long relative to estimated processing time,
        # fall back to single process
        estimated_processing_time = len(all_entries) * 0.0001  # rough estimate: 0.1ms per entry
        if load_time > estimated_processing_time * 2 and len(all_entries) < 20000:
            if verbose:
                print(f"Load time ({load_time:.3f}s) suggests single-process would be faster")
            use_multiprocessing = False
        else:
            if verbose:
                print(f"Processing {len(all_entries)} entries in chunks of {chunk_size}")
            
            # Split entries into chunks
            chunks = _chunk_fasta_entries(all_entries, chunk_size)
            
            # Prepare arguments for worker processes
            chunk_args = [
                (chunk, min_length, max_length, validate_sequences, gene_only)
                for chunk in chunks
            ]
            
            # Process chunks in parallel using fork (faster) with spawn fallback
            try:
                # Try fork first (much faster on Linux/Unix)
                try:
                    with Pool(processes=n_processes) as pool:
                        results = pool.map(_process_chunk, chunk_args)
                except (RuntimeError, OSError):
                    # Fallback to spawn if fork fails
                    if verbose:
                        print("Fork failed, using spawn context...")
                    ctx = get_context('spawn')
                    with ctx.Pool(processes=n_processes) as pool:
                        results = pool.map(_process_chunk, chunk_args)
                
                # Combine results from all processes
                processed_entries = []
                skipped_counts = {
                    'invalid_sequence': 0,
                    'length_filter': 0,
                    'gene_filter': 0,
                    'molecular_weight_error': 0
                }
                
                for chunk_entries, chunk_skipped in results:
                    processed_entries.extend(chunk_entries)
                    for key, count in chunk_skipped.items():
                        skipped_counts[key] += count
            
            except Exception as e:
                if verbose:
                    print(f"Multiprocessing failed, falling back to single process: {e}")
                use_multiprocessing = False
    
    if not use_multiprocessing:
        # Single-process fallback
        processed_entries = []
        skipped_counts = {
            'invalid_sequence': 0,
            'length_filter': 0,
            'gene_filter': 0,
            'molecular_weight_error': 0
        }
        
        for header, sequence in read_fasta_entries(fasta_path, verbose=False):
            try:
                entry_data, skip_reason = _process_single_entry(
                    header, sequence, min_length, max_length, 
                    validate_sequences, gene_only, verbose
                )
                
                if entry_data:
                    processed_entries.append(entry_data)
                elif skip_reason:
                    skipped_counts[skip_reason] += 1
                
            except Exception as e:
                if verbose:
                    print(f"Unexpected error processing entry {header[:50]}...: {e}")
                continue
    
    # Log processing statistics
    if verbose:
        total_processed = len(processed_entries)
        total_skipped = sum(skipped_counts.values())
        print(f"Processed {total_processed} entries, skipped {total_skipped}")
        for reason, count in skipped_counts.items():
            if count > 0:
                print(f"  - {reason}: {count}")
    
    # Create DataFrame
    if not processed_entries:
        if verbose:
            print("Warning: No valid sequences found in the file")
        return pd.DataFrame(columns=column_order)
    
    df = pd.DataFrame(processed_entries)
    
    # Select and order columns
    available_columns = [col for col in column_order if col in df.columns]
    if available_columns:
        df = df[available_columns]
    
    # Sort DataFrame
    if sort_by:
        valid_sort_columns = [col for col in sort_by if col in df.columns]
        if valid_sort_columns:
            valid_ascending = sort_ascending[:len(valid_sort_columns)]
            df = df.sort_values(by=valid_sort_columns, ascending=valid_ascending)
    
    if verbose:
        print(f"Generated DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    return df.reset_index(drop=True)
