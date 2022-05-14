import pandas as pd
import tenseal as ts
import os

dir = os.getcwd()
if dir.split("/")[-3] == "codebase":
    os.chdir("../../")


def context_gen():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    print("Is the context private?", ("Yes" if context.is_private() else "No"))
    print("Automatic relinearization is:", ("on" if context.auto_relin else "off"))
    print("Automatic rescaling is:", ("on" if context.auto_rescale else "off"))
    print(
        "Automatic modulus switching is:", ("on" if context.auto_mod_switch else "off")
    )
    return context


key_context = context_gen()
