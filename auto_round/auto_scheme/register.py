
AUTO_SCHEME_METHODS = {}

def register_scheme_methods(names):
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
                AUTO_SCHEME_METHODS[name] = alg
        else:
            AUTO_SCHEME_METHODS[names] = alg

        return alg

    return register