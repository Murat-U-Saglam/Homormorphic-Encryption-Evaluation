import tenseal as ts

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

v1 = [0, 1, 2, 3, 4]
v2 = [4, 3, 2, 1, 0]

# encrypted vectors
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)


print(dir(enc_v2))
print(enc_v2.serialize())
result = enc_v1 + enc_v2
result.decrypt() # ~ [4, 4, 4, 4, 4]

result = enc_v1.dot(enc_v2)
result.decrypt() # ~ [10]

matrix = [
  [73, 0.5, 8],
  [81, -5, 66],
  [-100, -78, -2],
  [0, 9, 17],
  [69, 11 , 10],
]
result = enc_v1.matmul(matrix)
a = result.decrypt() # ~ [157, -90, 153]
print(a)