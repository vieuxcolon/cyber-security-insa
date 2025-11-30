# ============================================================
#                    DES + Frequency Analysis
#              Full Experiment (Automatic Text Download)
# ============================================================

import requests
import string
import time
import os
import matplotlib.pyplot as plt
from Crypto.Cipher import DES
from typing import List

# ------------------------------------------------------------
# Step 1: Automatically download a large English text
# ------------------------------------------------------------

def download_large_text():
    url = "https://www.gutenberg.org/files/11/11-0.txt"   # Alice in Wonderland (public domain)
    print("Downloading large English text from Project Gutenberg...")
    text = requests.get(url).text
    print("Download complete. Length:", len(text), "characters")
    return text


# ------------------------------------------------------------
# Step 2: Simple Substitution (Caesar Cipher)
# ------------------------------------------------------------

def caesar_encrypt(text, shift=3):
    alphabet = string.ascii_lowercase
    encrypted = []

    for ch in text.lower():
        if ch in alphabet:
            encrypted.append(alphabet[(alphabet.index(ch) + shift) % 26])
        else:
            encrypted.append(ch)

    return "".join(encrypted)


# ------------------------------------------------------------
# Step 3: Frequency Analysis
# ------------------------------------------------------------

def frequency_analysis(text):
    freq = {c: 0 for c in string.ascii_lowercase}
    total = 0

    for ch in text.lower():
        if ch in freq:
            freq[ch] += 1
            total += 1

    for k in freq:
        freq[k] = freq[k] / total if total > 0 else 0

    return freq

def plot_frequency(freq_dict, title):
    letters = list(freq_dict.keys())
    values = list(freq_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(letters, values)
    plt.title(title)
    plt.xlabel("Letter")
    plt.ylabel("Frequency")
    plt.grid(True, axis="y")
    plt.show()


# ------------------------------------------------------------
# Step 4 — Full DES Implementation
# ------------------------------------------------------------

# ------------------ DES Tables ------------------
IP = [
    58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,
    62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,
    57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,
    61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7
]

FP = [
    40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,
    38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,
    36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,
    34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25
]

E = [
    32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,
    12,13,14,15,16,17,16,17,18,19,20,21,
    20,21,22,23,24,25,24,25,26,27,28,29,
    28,29,30,31,32,1
]

P = [
    16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,
    2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25
]

PC_1 = [
    57,49,41,33,25,17,9,1,58,50,42,34,26,18,
    10,2,59,51,43,35,27,19,11,3,60,52,44,36,
    63,55,47,39,31,23,15,7,62,54,46,38,30,22,
    14,6,61,53,45,37,29,21,13,5,28,20,12,4
]

PC_2 = [
    14,17,11,24,1,5,3,28,15,6,21,10,23,19,12,4,
    26,8,16,7,27,20,13,2,41,52,31,37,47,55,
    30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32
]

SHIFT_SCHEDULE = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

S_BOX = [
    # S1
    [[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
     [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
     [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
     [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]],
    # S2
    [[15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
     [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
     [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
     [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]],
    # S3
    [[10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
     [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
     [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
     [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]],
    # S4
    [[7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
     [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
     [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
     [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]],
    # S5
    [[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
     [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
     [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
     [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]],
    # S6
    [[12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
     [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
     [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
     [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]],
    # S7
    [[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
     [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
     [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
     [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]],
    # S8
    [[13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
     [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
     [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
     [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]
]

# ------------------ Helper Functions ------------------

def _to_bitlist(x: int, bits: int) -> List[int]:
    return [(x >> (bits - 1 - i)) & 1 for i in range(bits)]

def _from_bitlist(b: List[int]) -> int:
    x = 0
    for bit in b:
        x = (x << 1) | bit
    return x

def permute(block_bits: List[int], table: List[int]) -> List[int]:
    return [block_bits[i - 1] for i in table]

def left_rotate(bits: List[int], n: int) -> List[int]:
    return bits[n:] + bits[:n]

# ------------------ Key Schedule ------------------

def generate_round_keys(key64: bytes) -> List[List[int]]:
    key_int = int.from_bytes(key64, byteorder="big")
    key_bits = _to_bitlist(key_int, 64)
    permuted = permute(key_bits, PC_1)
    C, D = permuted[:28], permuted[28:]

    round_keys = []
    for shift in SHIFT_SCHEDULE:
        C = left_rotate(C, shift)
        D = left_rotate(D, shift)
        CD = C + D
        round_keys.append(permute(CD, PC_2))

    return round_keys

# ------------------ Round Function ------------------

def f_function(R_bits: List[int], round_key_bits: List[int]) -> List[int]:
    E_bits = permute(R_bits, E)
    xored = [b ^ k for b, k in zip(E_bits, round_key_bits)]

    out_bits = []
    for i in range(8):
        block6 = xored[i*6:(i+1)*6]
        row = (block6[0] << 1) | block6[5]
        col = (block6[1] << 3) | (block6[2] << 2) | (block6[3] << 1) | block6[4]
        s_val = S_BOX[i][row][col]
        out_bits.extend([(s_val >> (3 - j)) & 1 for j in range(4)])

    return permute(out_bits, P)

# ------------------ Encrypt/Decrypt ------------------

def des_encrypt_block(block8: bytes, round_keys):
    bits = _to_bitlist(int.from_bytes(block8, "big"), 64)
    bits = permute(bits, IP)

    L, R = bits[:32], bits[32:]

    for rk in round_keys:
        f_out = f_function(R, rk)
        L, R = R, [l ^ f for l, f in zip(L, f_out)]

    final_bits = permute(R + L, FP)
    return _from_bitlist(final_bits).to_bytes(8, "big")

def des_encrypt_message(msg: bytes, round_keys):
    if len(msg) % 8 != 0:
        msg += b"\x00" * (8 - len(msg) % 8)

    blocks = [msg[i:i+8] for i in range(0, len(msg), 8)]
    return b"".join(des_encrypt_block(b, round_keys) for b in blocks)

if __name__ == "__main__":
    # Step 1 — large text
    text = download_large_text()

    # Step 2 — Caesar encryption and frequency analysis (store results, no print yet)
    caesar_ciphertext = caesar_encrypt(text, shift=3)
    caesar_freq = frequency_analysis(caesar_ciphertext)

    # Plot Caesar frequencies
    plot_frequency(caesar_freq, "Letter Frequency After Caesar Cipher")

    # Step 3 — DES encryption and frequency analysis
    key = b"8bytekey"
    round_keys = generate_round_keys(key)
    des_ciphertext = des_encrypt_message(text.encode("utf-8"), round_keys)
    des_freq = frequency_analysis(des_ciphertext.decode("latin1"))

    # Plot DES frequencies
    plot_frequency(des_freq, "Letter Frequency After DES Cipher")

    # Step 4 — Print side-by-side comparison in terminal
    print("\nLetter Frequencies Comparison (Caesar vs DES):")
    print(f"{'Letter':^6} | {'Caesar':^10} | {'DES':^10}")
    print("-"*32)
    for letter in string.ascii_lowercase:
        caesar_val = f"{caesar_freq.get(letter,0):.4f}"
        des_val = f"{des_freq.get(letter,0):.4f}"
        print(f"{letter:^6} | {caesar_val:^10} | {des_val:^10}")

    print("\nExperiment completed.")


