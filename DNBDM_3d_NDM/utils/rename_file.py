import matplotlib.pyplot as plt
from gmssl.sm4 import CryptSM4, SM4_ENCRYPT

def hamming_distance(b1, b2):
    return sum(bin(x ^ y).count('1') for x, y in zip(b1, b2))

key = bytes.fromhex('0123456789abcdeffedcba9876543210')
plain = bytes.fromhex('0123456789abcdeffedcba9876543210')
sm4 = CryptSM4()
sm4.set_key(key, SM4_ENCRYPT)
base_cipher = sm4.crypt_ecb(plain)

diffs = []
for bit in range(128):
    p = bytearray(plain)
    byte_index, bit_index = divmod(bit, 8)
    p[byte_index] ^= (1 << (7 - bit_index))
    new_cipher = sm4.crypt_ecb(bytes(p))
    diffs.append(hamming_distance(base_cipher, new_cipher))

print(f'平均变化比特数: {sum(diffs)/len(diffs):.2f} / 128')
plt.bar(range(128), diffs)
plt.axhline(64, color='r', linestyle='--', label='理想值 64')
plt.xlabel('翻转比特位置')
plt.ylabel('密文变化比特数')
plt.title('SM4 雪崩效应实验')
plt.legend()
plt.show()
