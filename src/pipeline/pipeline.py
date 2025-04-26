"""
Pipeline module
"""
class Pipeline:
    """
    SfM pipeline
    """
    def __init__(self, *steps):
        self.steps = steps

    def _run(self):
        for step in self.steps:
            data = step()
        return data

    def __call__(self):
        return self._run()
