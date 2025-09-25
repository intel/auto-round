AUTO_SCHEMES_ALGS = {}

def register_dtype(names):
    """Class decorator to register a mixed precision algorithm to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        names: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(alg):
        if isinstance(names, (tuple, list)):
            for name in names:
                AUTO_SCHEMES_ALGS[name] = alg
        else:
            AUTO_SCHEMES_ALGS[names] = alg

        return alg

    return register
