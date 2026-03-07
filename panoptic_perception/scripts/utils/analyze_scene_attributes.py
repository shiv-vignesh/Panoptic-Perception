
import json
import os
from tqdm import tqdm

from collections import defaultdict

def analyze_scene_attributes(labels_dir:str, split:str):
    
    assert os.path.exists(f'{labels_dir}/{split}'), f'Path {labels_dir}/{split} does not exists!'
        
    source_dir = f'{labels_dir}/{split}'
    attributes_info = defaultdict(lambda : defaultdict(set))
    attributes_counts = defaultdict(lambda : defaultdict(int))

    total_occurance = len(os.listdir(source_dir))
    
    for label_json in tqdm(os.listdir(source_dir)):
        labels_dict = json.load(open(f'{source_dir}/{label_json}'))
        attributes = labels_dict.get("attributes", None)
        
        if attributes is not None:
            assert type(attributes) == dict, f'Expected Attributes to be a dict, got: {type(attributes)}'
            
            for k, v in attributes.items():
                attributes_info[k]["unique_values"].add(v)
                attributes_counts[k][v] += 1

    print("\n" + "=" * 70)
    print(f"SCENE ATTRIBUTES DISTRIBUTION ({split})")
    print("=" * 70)

    for attribute in attributes_info:
        print(f"\n{'Attribute:':<12} {attribute}")
        print(f"{'Value':<25} {'Count':<10} {'%':<10}")
        print("-" * 45)

        for value in sorted(attributes_counts[attribute], key=attributes_counts[attribute].get, reverse=True):
            count = attributes_counts[attribute][value]
            pct = count / total_occurance * 100
            print(f"{value:<25} {count:<10} {pct:.1f}%")

        print("-" * 45)
    
if __name__ == "__main__":
    
    analyze_scene_attributes(
        labels_dir="panoptic_perception/BDD100k/bdd100k_labels/100k",
        split="test"
    )