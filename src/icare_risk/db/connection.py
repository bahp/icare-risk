from sqlalchemy import create_engine

# Private global variable to hold the engine state
_global_engine = None


def setup_database(db_url: str, **engine_kwargs):
    """
    Initializes the global SQLAlchemy engine for the package.
    Call this once at the start of your script or pipeline.

    Args:
        db_url (str): The database connection string.
        **engine_kwargs: Extra arguments for create_engine (e.g., pool_size).
    """
    global _global_engine
    _global_engine = create_engine(db_url, **engine_kwargs)
    print("icare_risk: Database engine successfully configured.")


def get_engine():
    """
    Retrieves the global engine. Raises an error if it hasn't been set.
    """
    if _global_engine is None:
        raise RuntimeError(
            "Database engine is not configured! "
            "Please call `icare_risk.db.setup_database(db_url)` before querying data."
        )
    return _global_engine