from pipeline.element import BaseElement
import pycolmap

class ColmapSiftMatcher(BaseElement):
    """
    Matcher using COLMAP's SIFT features.
    """
    def __init__(self, db_path):
        super().__init__("Colmap SiFT Matcher")
        self.db_path = db_path

    def _process(self):
        """
        Process the data.
        """
        pycolmap.match_exhaustive(self.db_path)
