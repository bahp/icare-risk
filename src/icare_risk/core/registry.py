PHENOTYPE_REGISTRY = {}

def register_phenotype(name: str):
    """Decorator to register a phenotype function in the global pipeline."""
    def decorator(func):
        PHENOTYPE_REGISTRY[name] = func
        return func
    return decorator