from pipeline.element import BaseElement
import pycolmap

class ColmapSiftExtractor(BaseElement):
    """
    Extract SIFT features using COLMAP.
    """
    def __init__(self, path, db_path):
        super().__init__("Colmap SiFT Extractor")
        self.path = path
        self.db_path = db_path

    def _process(self):
        """
        Process the data.
        """
        pycolmap.extract_features(self.db_path, self.path)
