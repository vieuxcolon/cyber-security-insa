import os
import time
from collections import Counter
from Crypto.Cipher import DES

# ============================================================
#                  FEISTEL CIPHER USING XOR
# ============================================================

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def feistel_round(L, R, K):
    return R, xor_bytes(L, K)

def feistel_encrypt(plaintext, key, rounds=4):
    if len(plaintext) % 2 != 0:
        plaintext += b'\0'

    half = len(plaintext) // 2
    L = plaintext[:half]
    R = plaintext[half:]

    for _ in range(rounds):
        L, R = feistel_round(L, R, key)

    return R + L

def feistel_decrypt(cipher, key, rounds=4):
    half = len(cipher) // 2
    R = cipher[:half]
    L = cipher[half:]

    for _ in range(rounds):
        L, R = feistel_round(L, R, key)

    return L + R

# ============================================================
#                  TEST 1: FEISTEL TESTING
# ============================================================

def test_feistel():
    print("\n[INFO] Testing Feistel cipher...\n")

    key = os.urandom(8)
    messages = [
        b"HelloWorld",
        b"ABCDEF12",
        b"Crypto",
        b"Feistel123",
        os.urandom(16)
    ]

    for msg in messages:
        print(f"[INFO] Encrypting message: {msg}")
        cipher = feistel_encrypt(msg, key)
        print(f"[INFO] Ciphertext: {cipher.hex()}")
        plain = feistel_decrypt(cipher, key)
        print(f"[INFO] Decrypted: {plain}\n")
        assert plain.startswith(msg), "[ERROR] Feistel decryption mismatch!"

    print("[INFO] Feistel test completed successfully.\n")

# ============================================================
#                     MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    test_feistel()

