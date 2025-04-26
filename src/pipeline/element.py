"""

"""
import pycolmap
class BaseElement:
    """
    Base pipeline element class.
    """
    def __init__(self, name: str):
        self.name = name

    def _process(self):
        """
        Process the data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self):
        """
        Call the process method.
        """
        return self._process()

    def get_name(self):
        """
        Get the name of the element.
        """
        return self.name

class MapperElement(BaseElement):
    """
    Mapper element class.
    """
    def __init__(self, name: str, db_path: str):
        super().__init__(name)
        self.bundle_adjustment = None
        self.db_path = db_path

    def _process(self):
        """
        Process the data.
        """
        db = pycolmap.Database(self.db_path)
        cache = pycolmap.DatabaseCache.create(db)
        mapper = pycolmap.IncrementalMapper(cache)
        recon = pycolmap.Reconstruction()

        return recon
