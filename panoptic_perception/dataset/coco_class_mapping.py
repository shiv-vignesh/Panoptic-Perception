"""COCO 2017 class mapping utilities."""

from enum import Enum
from typing import Optional


class COCOCategories(Enum):
    """COCO 2017 detection categories (80 total, non-contiguous IDs 1-90)."""
    PERSON = 1
    BICYCLE = 2
    CAR = 3
    MOTORCYCLE = 4
    AIRPLANE = 5
    BUS = 6
    TRAIN = 7
    TRUCK = 8
    BOAT = 9
    TRAFFIC_LIGHT = 10
    FIRE_HYDRANT = 11
    STOP_SIGN = 13
    PARKING_METER = 14
    BENCH = 15
    BIRD = 16
    CAT = 17
    DOG = 18
    HORSE = 19
    SHEEP = 20
    COW = 21
    ELEPHANT = 22
    BEAR = 23
    ZEBRA = 24
    GIRAFFE = 25
    BACKPACK = 27
    UMBRELLA = 28
    HANDBAG = 31
    TIE = 32
    SUITCASE = 33
    FRISBEE = 34
    SKIS = 35
    SNOWBOARD = 36
    SPORTS_BALL = 37
    KITE = 38
    BASEBALL_BAT = 39
    BASEBALL_GLOVE = 40
    SKATEBOARD = 41
    SURFBOARD = 42
    TENNIS_RACKET = 43
    BOTTLE = 44
    WINE_GLASS = 46
    CUP = 47
    FORK = 48
    KNIFE = 49
    SPOON = 50
    BOWL = 51
    BANANA = 52
    APPLE = 53
    SANDWICH = 54
    ORANGE = 55
    BROCCOLI = 56
    CARROT = 57
    HOT_DOG = 58
    PIZZA = 59
    DONUT = 60
    CAKE = 61
    CHAIR = 62
    COUCH = 63
    POTTED_PLANT = 64
    BED = 65
    DINING_TABLE = 67
    TOILET = 70
    TV = 72
    LAPTOP = 73
    MOUSE = 74
    REMOTE = 75
    KEYBOARD = 76
    CELL_PHONE = 77
    MICROWAVE = 78
    OVEN = 79
    TOASTER = 80
    SINK = 81
    REFRIGERATOR = 82
    BOOK = 84
    CLOCK = 85
    VASE = 86
    SCISSORS = 87
    TEDDY_BEAR = 88
    HAIR_DRIER = 89
    TOOTHBRUSH = 90

    @classmethod
    def from_id(cls, class_id: int) -> Optional[str]:
        """Get lowercase class name from COCO category id."""
        try:
            return cls(class_id).name.lower()
        except ValueError:
            return None

    @classmethod
    def from_label(cls, label: str) -> int:
        """Get COCO category id from label name (space-tolerant)."""
        return cls[label.upper().replace(" ", "_")].value


class COCOSupercategories(Enum):
    """COCO 2017 supercategories (12 groups). IDs assigned by order of appearance."""
    PERSON = 0
    VEHICLE = 1
    OUTDOOR = 2
    ANIMAL = 3
    ACCESSORY = 4
    SPORTS = 5
    KITCHEN = 6
    FOOD = 7
    FURNITURE = 8
    ELECTRONIC = 9
    APPLIANCE = 10
    INDOOR = 11

    @classmethod
    def from_id(cls, class_id: int) -> Optional[str]:
        """Get lowercase supercategory name from id."""
        try:
            return cls(class_id).name.lower()
        except ValueError:
            return None

    @classmethod
    def from_label(cls, label: str) -> int:
        """Get supercategory id from label name."""
        return cls[label.upper()].value


CATEGORY_TO_SUPERCATEGORY = {
    COCOCategories.PERSON: COCOSupercategories.PERSON,
    COCOCategories.BICYCLE: COCOSupercategories.VEHICLE,
    COCOCategories.CAR: COCOSupercategories.VEHICLE,
    COCOCategories.MOTORCYCLE: COCOSupercategories.VEHICLE,
    COCOCategories.AIRPLANE: COCOSupercategories.VEHICLE,
    COCOCategories.BUS: COCOSupercategories.VEHICLE,
    COCOCategories.TRAIN: COCOSupercategories.VEHICLE,
    COCOCategories.TRUCK: COCOSupercategories.VEHICLE,
    COCOCategories.BOAT: COCOSupercategories.VEHICLE,
    COCOCategories.TRAFFIC_LIGHT: COCOSupercategories.OUTDOOR,
    COCOCategories.FIRE_HYDRANT: COCOSupercategories.OUTDOOR,
    COCOCategories.STOP_SIGN: COCOSupercategories.OUTDOOR,
    COCOCategories.PARKING_METER: COCOSupercategories.OUTDOOR,
    COCOCategories.BENCH: COCOSupercategories.OUTDOOR,
    COCOCategories.BIRD: COCOSupercategories.ANIMAL,
    COCOCategories.CAT: COCOSupercategories.ANIMAL,
    COCOCategories.DOG: COCOSupercategories.ANIMAL,
    COCOCategories.HORSE: COCOSupercategories.ANIMAL,
    COCOCategories.SHEEP: COCOSupercategories.ANIMAL,
    COCOCategories.COW: COCOSupercategories.ANIMAL,
    COCOCategories.ELEPHANT: COCOSupercategories.ANIMAL,
    COCOCategories.BEAR: COCOSupercategories.ANIMAL,
    COCOCategories.ZEBRA: COCOSupercategories.ANIMAL,
    COCOCategories.GIRAFFE: COCOSupercategories.ANIMAL,
    COCOCategories.BACKPACK: COCOSupercategories.ACCESSORY,
    COCOCategories.UMBRELLA: COCOSupercategories.ACCESSORY,
    COCOCategories.HANDBAG: COCOSupercategories.ACCESSORY,
    COCOCategories.TIE: COCOSupercategories.ACCESSORY,
    COCOCategories.SUITCASE: COCOSupercategories.ACCESSORY,
    COCOCategories.FRISBEE: COCOSupercategories.SPORTS,
    COCOCategories.SKIS: COCOSupercategories.SPORTS,
    COCOCategories.SNOWBOARD: COCOSupercategories.SPORTS,
    COCOCategories.SPORTS_BALL: COCOSupercategories.SPORTS,
    COCOCategories.KITE: COCOSupercategories.SPORTS,
    COCOCategories.BASEBALL_BAT: COCOSupercategories.SPORTS,
    COCOCategories.BASEBALL_GLOVE: COCOSupercategories.SPORTS,
    COCOCategories.SKATEBOARD: COCOSupercategories.SPORTS,
    COCOCategories.SURFBOARD: COCOSupercategories.SPORTS,
    COCOCategories.TENNIS_RACKET: COCOSupercategories.SPORTS,
    COCOCategories.BOTTLE: COCOSupercategories.KITCHEN,
    COCOCategories.WINE_GLASS: COCOSupercategories.KITCHEN,
    COCOCategories.CUP: COCOSupercategories.KITCHEN,
    COCOCategories.FORK: COCOSupercategories.KITCHEN,
    COCOCategories.KNIFE: COCOSupercategories.KITCHEN,
    COCOCategories.SPOON: COCOSupercategories.KITCHEN,
    COCOCategories.BOWL: COCOSupercategories.KITCHEN,
    COCOCategories.BANANA: COCOSupercategories.FOOD,
    COCOCategories.APPLE: COCOSupercategories.FOOD,
    COCOCategories.SANDWICH: COCOSupercategories.FOOD,
    COCOCategories.ORANGE: COCOSupercategories.FOOD,
    COCOCategories.BROCCOLI: COCOSupercategories.FOOD,
    COCOCategories.CARROT: COCOSupercategories.FOOD,
    COCOCategories.HOT_DOG: COCOSupercategories.FOOD,
    COCOCategories.PIZZA: COCOSupercategories.FOOD,
    COCOCategories.DONUT: COCOSupercategories.FOOD,
    COCOCategories.CAKE: COCOSupercategories.FOOD,
    COCOCategories.CHAIR: COCOSupercategories.FURNITURE,
    COCOCategories.COUCH: COCOSupercategories.FURNITURE,
    COCOCategories.POTTED_PLANT: COCOSupercategories.FURNITURE,
    COCOCategories.BED: COCOSupercategories.FURNITURE,
    COCOCategories.DINING_TABLE: COCOSupercategories.FURNITURE,
    COCOCategories.TOILET: COCOSupercategories.FURNITURE,
    COCOCategories.TV: COCOSupercategories.ELECTRONIC,
    COCOCategories.LAPTOP: COCOSupercategories.ELECTRONIC,
    COCOCategories.MOUSE: COCOSupercategories.ELECTRONIC,
    COCOCategories.REMOTE: COCOSupercategories.ELECTRONIC,
    COCOCategories.KEYBOARD: COCOSupercategories.ELECTRONIC,
    COCOCategories.CELL_PHONE: COCOSupercategories.ELECTRONIC,
    COCOCategories.MICROWAVE: COCOSupercategories.APPLIANCE,
    COCOCategories.OVEN: COCOSupercategories.APPLIANCE,
    COCOCategories.TOASTER: COCOSupercategories.APPLIANCE,
    COCOCategories.SINK: COCOSupercategories.APPLIANCE,
    COCOCategories.REFRIGERATOR: COCOSupercategories.APPLIANCE,
    COCOCategories.BOOK: COCOSupercategories.INDOOR,
    COCOCategories.CLOCK: COCOSupercategories.INDOOR,
    COCOCategories.VASE: COCOSupercategories.INDOOR,
    COCOCategories.SCISSORS: COCOSupercategories.INDOOR,
    COCOCategories.TEDDY_BEAR: COCOSupercategories.INDOOR,
    COCOCategories.HAIR_DRIER: COCOSupercategories.INDOOR,
    COCOCategories.TOOTHBRUSH: COCOSupercategories.INDOOR,
}


def supercategory_of(category_id: int) -> Optional[int]:
    """Given a COCO category id, return its supercategory id (or None if unknown)."""
    try:
        return CATEGORY_TO_SUPERCATEGORY[COCOCategories(category_id)].value
    except (ValueError, KeyError):
        return None


def categories_in_supercategory(supercategory_id: int) -> list:
    """Return list of COCO category ids belonging to the given supercategory id."""
    try:
        target = COCOSupercategories(supercategory_id)
    except ValueError:
        return []
    return [cat.value for cat, sup in CATEGORY_TO_SUPERCATEGORY.items() if sup is target]


if __name__ == "__main__":
    print(f"categories: {len(COCOCategories)}, supercategories: {len(COCOSupercategories)}")
    print(f"from_id(3): {COCOCategories.from_id(3)}")
    print(f"from_label('traffic light'): {COCOCategories.from_label('traffic light')}")
    print(f"supercategory_of(3): {COCOSupercategories.from_id(supercategory_of(3))}")
    print(f"categories_in_supercategory(VEHICLE=1): {categories_in_supercategory(1)}")
