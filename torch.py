class Generator:
    def __init__(self, device=None):
        pass
    def manual_seed(self, seed):
        return self

class backends:
    class mps:
        @staticmethod
        def is_available():
            return False

class cuda:
    @staticmethod
    def is_available():
        return False

float16 = 'float16'
float32 = 'float32'
