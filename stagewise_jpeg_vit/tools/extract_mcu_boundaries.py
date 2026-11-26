import sys
from pathlib import Path
import torch

# Ensure the repo root is in python path
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import byte_manip as bm

def get_jpeg_mcu_byte_boundaries(jpeg_bytes):
    """
    Given raw JPEG bytes (as a bytes object or list of uint8),
    returns the exact raw file byte index boundaries for each of the MCUs.
    
    Returns:
        boundaries: list of integers representing raw byte boundary indices.
                    For K MCUs, there are K+1 boundaries:
                    [start_0, start_1, ..., start_{K-1}, end_{K-1}]
    """
    # 1. Parse JPEG metadata and MCU bit ranges using pybind extension
    syntax = bm.read_syntax_patches_from_buffer(jpeg_bytes)
    header = syntax["header"]
    mcus = syntax["mcus"]
    
    entropy_start = header["entropy_start_offset"]
    entropy_end = header["entropy_end_offset"]
    
    # 2. Build the bit-to-raw-byte index mapping
    pos = entropy_start
    current_bit_offset = 0
    bit_to_byte_map = {}
    
    while pos < entropy_end:
        b = jpeg_bytes[pos]
        pos += 1
        if b == 0xFF:
            if pos >= entropy_end:
                break
            next_byte = jpeg_bytes[pos]
            if next_byte == 0x00:
                pos += 1
                # 0xFF00 byte maps to the two raw bytes (pos-2, pos-1).
                # We map all 8 bits to the final raw byte index of the token.
                for bit_i in range(8):
                    bit_to_byte_map[current_bit_offset + bit_i] = pos - 1
                current_bit_offset += 8
                continue
            if 0xD0 <= next_byte <= 0xD7:
                pos += 1
                # Restart marker, no payload bits.
                continue
            # End of entropy stream or other marker.
            break
        else:
            for bit_i in range(8):
                bit_to_byte_map[current_bit_offset + bit_i] = pos - 1
            current_bit_offset += 8
            
    # 3. Translate MCU bit offsets to raw byte offsets
    boundaries = []
    for i, mcu in enumerate(mcus):
        bit_start = mcu["bit_start"]
        bit_end = mcu["bit_end"]
        
        # Translate bit_start
        if bit_start in bit_to_byte_map:
            byte_start = bit_to_byte_map[bit_start]
        else:
            # Fallback for boundary conditions
            byte_start = entropy_start if i == 0 else boundaries[-1]
            
        # Translate bit_end (exclusive boundary)
        if (bit_end - 1) in bit_to_byte_map:
            byte_end = bit_to_byte_map[bit_end - 1] + 1
        else:
            byte_end = entropy_end if i == len(mcus) - 1 else byte_start
            
        # Ensure boundaries are monotonic
        if i > 0 and byte_start < boundaries[-1]:
            byte_start = boundaries[-1]
            
        if byte_end < byte_start:
            byte_end = byte_start
            
        if i == 0:
            boundaries.append(byte_start)
        boundaries.append(byte_end)
        
    return boundaries

if __name__ == "__main__":
    # Test script self-check
    print("mcu boundaries module loaded successfully.")
