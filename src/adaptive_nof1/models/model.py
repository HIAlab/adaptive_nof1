from src.adaptive_nof1.basic_types import Observation


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_context(self) -> Dict:
        pass

    @abstractmethod
    def observe_outcome(self, action, context) -> Observation:
        pass
