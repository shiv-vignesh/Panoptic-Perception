"""
BDD100K class mapping utilities for reducing 10 classes to 6 merged classes.
This merges similar classes to improve training convergence.
"""

from enum import Enum
from typing import Optional


class BDD100KClassesOriginal(Enum):
    """Original 10 BDD100K detection classes."""
    PEDESTRIAN = 0
    RIDER = 1
    CAR = 2
    TRUCK = 3
    BUS = 4
    TRAIN = 5
    MOTORCYCLE = 6
    BICYCLE = 7
    TRAFFIC_LIGHT = 8
    TRAFFIC_SIGN = 9


class BDD100KClassesReduced(Enum):
    """Reduced 6 BDD100K classes by merging similar objects."""
    PERSON = 0          # pedestrian
    RIDER = 1           # rider
    VEHICLE = 2         # car, truck, bus, train
    TWO_WHEELER = 3     # motorcycle, bicycle
    TRAFFIC_LIGHT = 4   # traffic_light
    TRAFFIC_SIGN = 5    # traffic_sign

    @classmethod
    def from_id(cls, class_id: int) -> Optional[str]:
        """Get class name from ID."""
        try:
            return cls(class_id).name.lower()
        except ValueError:
            return None

    @classmethod
    def from_label(cls, label: str) -> int:
        """Get class ID from label name."""
        return cls[label.upper()].value


# Mapping from original 10 classes to reduced 6 classes
ORIGINAL_TO_REDUCED_MAPPING = {
    0: 0,  # PEDESTRIAN -> PERSON
    1: 1,  # RIDER -> RIDER
    2: 2,  # CAR -> VEHICLE
    3: 2,  # TRUCK -> VEHICLE
    4: 2,  # BUS -> VEHICLE
    5: 2,  # TRAIN -> VEHICLE
    6: 3,  # MOTORCYCLE -> TWO_WHEELER
    7: 3,  # BICYCLE -> TWO_WHEELER
    8: 4,  # TRAFFIC_LIGHT -> TRAFFIC_LIGHT
    9: 5,  # TRAFFIC_SIGN -> TRAFFIC_SIGN
}

# Reverse mapping for reference
REDUCED_TO_ORIGINAL_MAPPING = {
    0: [0],           # PERSON <- pedestrian
    1: [1],           # RIDER <- rider
    2: [2, 3, 4, 5],  # VEHICLE <- car, truck, bus, train
    3: [6, 7],        # TWO_WHEELER <- motorcycle, bicycle
    4: [8],           # TRAFFIC_LIGHT <- traffic_light
    5: [9],           # TRAFFIC_SIGN <- traffic_sign
}

# Class name mapping
ORIGINAL_CLASS_NAMES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

REDUCED_CLASS_NAMES = [
    'person', 'rider', 'vehicle', 'two_wheeler', 'traffic_light', 'traffic_sign'
]


def map_original_to_reduced(original_class_id: int) -> int:
    """
    Map original BDD100K class ID (0-9) to reduced class ID (0-5).

    Args:
        original_class_id: Original class ID (0-9)

    Returns:
        Reduced class ID (0-5)

    Raises:
        ValueError: If original_class_id is not in range [0, 9]
    """
    if original_class_id not in ORIGINAL_TO_REDUCED_MAPPING:
        raise ValueError(f"Invalid class ID: {original_class_id}. Must be in range [0, 9]")

    return ORIGINAL_TO_REDUCED_MAPPING[original_class_id]


def map_reduced_to_original(reduced_class_id: int) -> list:
    """
    Get all original class IDs that map to a reduced class ID.

    Args:
        reduced_class_id: Reduced class ID (0-5)

    Returns:
        List of original class IDs

    Raises:
        ValueError: If reduced_class_id is not in range [0, 5]
    """
    if reduced_class_id not in REDUCED_TO_ORIGINAL_MAPPING:
        raise ValueError(f"Invalid reduced class ID: {reduced_class_id}. Must be in range [0, 5]")

    return REDUCED_TO_ORIGINAL_MAPPING[reduced_class_id]


def get_original_class_name(class_id: int) -> str:
    """Get original class name from ID."""
    if 0 <= class_id < len(ORIGINAL_CLASS_NAMES):
        return ORIGINAL_CLASS_NAMES[class_id]
    raise ValueError(f"Invalid class ID: {class_id}")


def get_reduced_class_name(class_id: int) -> str:
    """Get reduced class name from ID."""
    if 0 <= class_id < len(REDUCED_CLASS_NAMES):
        return REDUCED_CLASS_NAMES[class_id]
    raise ValueError(f"Invalid class ID: {class_id}")


def print_mapping_info():
    """Print detailed mapping information."""
    print("="*70)
    print("BDD100K CLASS MAPPING: 10 Classes -> 6 Classes")
    print("="*70)
    print()

    for reduced_id in range(6):
        reduced_name = REDUCED_CLASS_NAMES[reduced_id]
        original_ids = REDUCED_TO_ORIGINAL_MAPPING[reduced_id]
        original_names = [ORIGINAL_CLASS_NAMES[i] for i in original_ids]

        print(f"Class {reduced_id}: {reduced_name.upper()}")
        print(f"  <- {', '.join(original_names)}")
        print()

    print("="*70)
    print()

    # Show mapping table
    print("DETAILED MAPPING TABLE:")
    print("-"*70)
    print(f"{'Original ID':<15} {'Original Name':<20} -> {'Reduced ID':<15} {'Reduced Name':<20}")
    print("-"*70)

    for orig_id in range(10):
        orig_name = ORIGINAL_CLASS_NAMES[orig_id]
        reduced_id = ORIGINAL_TO_REDUCED_MAPPING[orig_id]
        reduced_name = REDUCED_CLASS_NAMES[reduced_id]
        print(f"{orig_id:<15} {orig_name:<20} -> {reduced_id:<15} {reduced_name:<20}")

    print("-"*70)


if __name__ == "__main__":
    # Test the mapping
    print_mapping_info()

    # Example usage
    print("\nEXAMPLE USAGE:")
    print("-"*70)

    # Map original class to reduced
    print(f"Original class 2 (car) -> Reduced class {map_original_to_reduced(2)} (vehicle)")
    print(f"Original class 3 (truck) -> Reduced class {map_original_to_reduced(3)} (vehicle)")
    print(f"Original class 6 (motorcycle) -> Reduced class {map_original_to_reduced(6)} (two_wheeler)")

    print()

    # Map reduced class to original
    print(f"Reduced class 2 (vehicle) <- Original classes {map_reduced_to_original(2)}")
    print(f"Reduced class 3 (two_wheeler) <- Original classes {map_reduced_to_original(3)}")

    print()

    # Using enum
    print(f"Reduced class name from ID 2: {BDD100KClassesReduced.from_id(2)}")
    print(f"Reduced class ID from label 'VEHICLE': {BDD100KClassesReduced.from_label('VEHICLE')}")
