from enum import Enum

class BDD100KClasses(Enum):
    PERSON = 0
    RIDER = 1
    CAR = 2
    TRUCK = 3
    BUS = 4
    TRAIN = 5
    MOTORCYCLE = 6
    BIKE = BICYCLE = 7
    TRAFFIC_LIGHT = 8
    TRAFFIC_SIGN = 9
    MOTOR = 10

    @classmethod
    def from_id(cls, class_id: int):
        try:
            return cls(class_id).name.lower()
        except ValueError:
            return None

    @classmethod
    def from_label(cls, label: str):
        return cls[label.upper()].value
