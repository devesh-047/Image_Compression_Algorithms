# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% [markdown]
# ## Huffman Compression Implementation

# %%
import numpy as np
from collections import Counter
import string
import matplotlib.pyplot as plt

# ---------------- Allowed symbols for N-ary Huffman ---------------- #
# Digits + uppercase + lowercase => 10 + 26 + 26 = 62 symbols
SYMBOLS = string.digits + string.ascii_uppercase + string.ascii_lowercase

# ---------------- Huffman Utilities ---------------- #
def build_nary_huffman(freqs, base):
    heap = [[f, [[sym, ""]]] for sym, f in freqs.items()]
    while len(heap) > 1:
        heap.sort(key=lambda x: x[0])
        children = heap[:base]
        rest = heap[base:]
        total_freq = sum(ch[0] for ch in children)

        newlist = []
        for i, child in enumerate(children):
            for sym, code in child[1]:
                newlist.append([sym, SYMBOLS[i] + code])

        heap = rest + [[total_freq, newlist]]
    return sorted(heap[0][1], key=lambda p: (len(p[-1]), p))

def get_huffman_codebook(freqs, base):
    huff_list = build_nary_huffman(freqs, base)
    return {sym: code for sym, code in huff_list}

# ---------------- Huffman Encode/Decode ---------------- #
def huffman_encode(img, base=2):
    flat = img.flatten()
    freqs = Counter(flat)
    codebook = get_huffman_codebook(freqs, base)

    encoded = [codebook[val] for val in flat]
    return encoded, codebook

def huffman_decode(encoded, codebook, shape, dtype):
    inv_codebook = {v: k for k, v in codebook.items()}
    decoded_vals = []
    buffer = ''
    for sym in encoded:
        buffer += sym
        if buffer in inv_codebook:
            decoded_vals.append(inv_codebook[buffer])
            buffer = ''
    return np.array(decoded_vals, dtype=dtype).reshape(shape)

# ---------------- Compression Ratio ---------------- #
def calculate_huffman_compression_ratio(img, base=2):
    flat = img.flatten()
    orig_bits = img.nbytes * 8  # Total bits in original image

    encoded, codebook = huffman_encode(img, base)

    encoded_bits = sum(len(sym) for sym in encoded) * 8  # SINCE N_MAX = 62, we can assume that each symbol is 1 byte, since that can support upto 256 unique symbols
    dict_bits = 0
    for k, v in codebook.items():
        key_bits = 32  # float or int
        value_bits = len(v) * 8  # 1 byte per symbol
        dict_bits += key_bits + value_bits

    total_bits = encoded_bits + dict_bits
    compression_ratio = orig_bits / total_bits

    return compression_ratio, orig_bits, encoded_bits, dict_bits, total_bits, codebook, encoded

def plot_compression_vs_N(img, max_base=62):
    """
    Computes and plots Huffman compression ratio for N-ary Huffman
    with N from 2 up to max_base.
    
    Parameters:
        img : np.ndarray
            Input image array (int or float).
        max_base : int
            Maximum N value to test (≤ length of SYMBOLS).
    """
    ratios = []
    bases = list(range(2, max_base + 1))
    
    flat = img.flatten()
    orig_bits = img.nbytes * 8
    
    for base in bases:
        freqs = Counter(flat)
        codebook = get_huffman_codebook(freqs, base)
        encoded = [codebook[val] for val in flat]

        # Estimate encoded data size
        encoded_bits = sum(len(sym) for sym in encoded) * 8

        # Estimate dictionary size
        dict_bits = 0
        for k, v in codebook.items():
            key_bits = 32  # assume float/int 32 bits
            value_bits = len(v) * 8
            dict_bits += key_bits + value_bits

        total_bits = encoded_bits + dict_bits
        compression_ratio = orig_bits / total_bits
        ratios.append(compression_ratio)
    
    plt.figure(figsize=(10, 5))
    plt.plot(bases, ratios, marker='o')
    plt.xlabel("N (N-ary Huffman)")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio vs N for N-ary Huffman")
    plt.grid(True)
    plt.show()





# %% [markdown]
# ## LZW Compression Implementation

# %%
import numpy as np
import matplotlib.pyplot as plt

def lzw_encode(image_data, max_dict_size=4096):
    """
    Encodes image data using the LZW algorithm.
    """
    flat_data = image_data.ravel().tobytes()
    
    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256
    
    encoded_data = []
    current_sequence = b""
    
    for byte in flat_data:
        current_byte = bytes([byte])
        new_sequence = current_sequence + current_byte
        
        if new_sequence in dictionary:
            current_sequence = new_sequence
        else:
            encoded_data.append(dictionary[current_sequence])
            if len(dictionary) < max_dict_size:
                dictionary[new_sequence] = next_code
                next_code += 1
            current_sequence = current_byte
            
    if current_sequence:
        encoded_data.append(dictionary[current_sequence])
        
    return encoded_data

def lzw_decode(encoded_data, shape,dtype):
    """
    Decodes LZW-encoded data back into an image array.
    """
    if not encoded_data:
        return np.zeros(shape, dtype=np.uint8)

    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256
    
    result = []
    prev_code = encoded_data.pop(0)
    current_entry = dictionary[prev_code]
    result.append(current_entry)
    
    for code in encoded_data:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = current_entry + current_entry[:1]
        else:
            raise ValueError("Invalid LZW code during decoding.")
            
        result.append(entry)
        
        new_entry = current_entry + entry[:1]
        dictionary[next_code] = new_entry
        next_code += 1
        
        current_entry = entry

    decoded_bytes = b"".join(result)
    decoded_image_array = np.frombuffer(decoded_bytes, dtype=dtype).reshape(shape)
    
    return decoded_image_array

def calculate_lzw_compression_ratio(original_data, encoded_data):
    """
    Calculates the compression ratio.
    """
    # Use the uint8 version for accurate byte size calculation
    original_size_bytes = original_data.size * original_data.itemsize
    
    # Size of encoded data: number of codes * size of a code in bytes
    # The LZW codes can have varying bit lengths. For a simple approximation,
    # we use the item size of the numpy array of codes.
    encoded_size_bytes = len(encoded_data) * np.array(encoded_data).itemsize
    
    if encoded_size_bytes == 0:
        return float('inf')
        
    return original_size_bytes / encoded_size_bytes


def test_lzw_dict_sizes(image_data, dict_sizes):
    """
    Tests LZW compression with different maximum dictionary sizes.
    
    Args:
        image_data (np.ndarray): The uint8 NumPy array of the image.
        dict_sizes (list): A list of integer maximum dictionary sizes to test.
        
    Returns:
        dict: A dictionary containing metrics for each max_dict_size.
    """
    results = {}
    
    for size in dict_sizes:
        print(f"Testing with max_dict_size = {size}...")
        
        # Encode the image
        encoded_codes = lzw_encode(image_data, max_dict_size=size)
        
        # Decode the image
        decoded_image_uint8 = lzw_decode(encoded_codes, image_data.shape,image_data.dtype)
        
        # Verify lossless compression
        is_equal = np.array_equal(image_data, decoded_image_uint8)
        
        # Calculate metrics
        orig_bits = image_data.size * 8
        encoded_bits = len(encoded_codes) * np.array(encoded_codes).itemsize * 8
        metadata_bits = len(image_data.shape) * 32  # Estimate for storing shape
        total_bits = encoded_bits + metadata_bits
        
        if total_bits > 0:
            compression_ratio = orig_bits / total_bits
        else:
            compression_ratio = float('inf')
        
        results[size] = {
            'compression_ratio': compression_ratio,
            'encoded_codes_len': len(encoded_codes),
            'is_lossless': is_equal
        }
        
    return results

def plot_lzw_results(results):
    """
    Plots the compression ratio vs. max dictionary size.
    
    Args:
        results (dict): The results from test_lzw_dict_sizes.
    """
    sizes = sorted(results.keys())
    ratios = [results[size]['compression_ratio'] for size in sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, ratios, marker='o')
    plt.title('LZW Compression Ratio vs. Max Dictionary Size')
    plt.xlabel('Max Dictionary Size')
    plt.ylabel('Compression Ratio')
    plt.xscale('log') # Use a log scale for dictionary size
    plt.grid(True)
    plt.xticks(sizes, labels=[str(s) for s in sizes])
    plt.show()


# ---------------- Run Length Encoding (RLE) ---------------- #

def rle_encode(img):
    flat = img.flatten()
    encoded = []
    
    if len(flat) == 0:
        return encoded, img.shape, img.dtype
    
    current_val = flat[0]
    count = 1
    
    for val in flat[1:]:
        if val == current_val:
            count += 1
        else:
            encoded.append((current_val, count))
            current_val = val
            count = 1
    
    # Append the last run
    encoded.append((current_val, count))
    
    return encoded, img.shape, img.dtype


def rle_decode(encoded, shape, dtype):
    decoded_vals = []
    
    for val, count in encoded:
        decoded_vals.extend([val] * count)
    
    return np.array(decoded_vals, dtype=dtype).reshape(shape)


# ---------------- Compression Ratio Calculation ---------------- #

def calculate_rle_compression_ratio(img):
    """
    Calculate compression ratio for RLE encoding.
    
    Args:
        img: input image (numpy array)
    
    Returns:
        tuple: (compression_ratio, orig_bits, encoded_bits, metadata_bits, 
                total_bits, encoded_data)
    """
    # Original size in bits
    orig_bits = img.nbytes * 8
    
    # Encode the image
    encoded, shape, dtype = rle_encode(img)
    
    # Calculate encoded data size
    # Each run is (value, count) where:
    # - value: same size as original dtype
    # - count: we'll use 32 bits (int32) to store counts
    bytes_per_pixel = img.dtype.itemsize
    encoded_bits = len(encoded) * (bytes_per_pixel * 8 + 32)
    
    # Metadata: shape (3 or 2 dimensions * 32 bits) + dtype info (32 bits)
    metadata_bits = len(shape) * 32 + 32
    
    total_bits = encoded_bits + metadata_bits
    compression_ratio = orig_bits / total_bits if total_bits > 0 else 0
    
    return compression_ratio, orig_bits, encoded_bits, metadata_bits, total_bits, encoded, shape, dtype

# %% [markdown]
# ## Run All Compression Algorithms

# %%

def test_all(img):
    # Ensure image is loaded
    if img is None:
        raise ValueError("Please load an image in the previous cell (variable name: img).")
    
    # Run Huffman Compression
    base = 52
    huffman_encoded, codebook = huffman_encode(img,base=base)
    huffman_decoded = huffman_decode(huffman_encoded,codebook,img.shape,img.dtype)
    huffman_ratio = calculate_huffman_compression_ratio(img,base=base)[0]
    
    # Run LZW Compression
    
    lzw_encoded = lzw_encode(img)
    lzw_decoded = lzw_decode(lzw_encoded, img.shape,img.dtype)
    lzw_ratio = calculate_lzw_compression_ratio(img, lzw_encoded)
    
    # Run RLE Compression
    rle_encoded, shape, dtype = rle_encode(img)
    rle_decoded = rle_decode(rle_encoded, shape, dtype)
    rle_ratio = calculate_rle_compression_ratio(img)[0]
    
    # Convert decoded images to displayable format (uint8)
    import numpy as np
    
    def to_uint8(img):
        """
        Converts an image to uint8, correctly handling
        both float ([0.0, 1.0]) and int ([0, 255]) data.
        """
        # Check if the image is a floating-point type
        if np.issubdtype(img.dtype, np.floating):
            # Assumes float data is in the [0.0, 1.0] range.
            # 1. Clip to handle any minor precision errors
            img = np.clip(img, 0.0, 1.0)
            
            # 2. Scale to [0, 255]
            img = img * 255.0
            
        # For both scaled floats and existing integers:
        # 1. Clip to the valid [0, 255] range
        # 2. Cast to uint8
        return np.clip(img, 0, 255).astype(np.uint8)
    
    huffman_img = to_uint8(huffman_decoded)
    lzw_img = to_uint8(lzw_decoded)
    rle_img = to_uint8(rle_decoded)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(huffman_img)
    axes[1].set_title(f"Huffman Compression (N=52)\nRatio: {huffman_ratio:.2f}")
    axes[1].axis("off")
    
    axes[2].imshow(lzw_img)
    axes[2].set_title(f"LZW Compression (dict_size = 4096) \nRatio: {lzw_ratio:.2f}")
    axes[2].axis("off")
    
    axes[3].imshow(rle_img)
    axes[3].set_title(f"RLE Compression\nRatio: {rle_ratio:.2f}")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.show()
    


