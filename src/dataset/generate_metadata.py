import os
import json
import pandas as pd
import random
import argparse
from tqdm import tqdm

def process_dataset(data_path, output_file="dataset.json"):

    data = {
        "train": [],
        "test": []
    }

    random.seed(42)

    for f_counter, folder in enumerate(sorted(os.listdir(data_path))):
        print(f"Start process {folder}")
        folder_path = os.path.join(data_path, folder)
        metadata_path = os.path.join(folder_path, "metadata.csv")

        meta = pd.read_csv(metadata_path)

        test_images_class_0 = []
        test_images_class_1 = []
        train_images = []

        for _, row in tqdm(meta.iterrows(), total=len(meta)):
            full_image_path = os.path.join(folder_path, row['image_path'])
            target = 1 if row['target'] != 0 else 0
            data_generator = folder

            image_data = {
                "image_path": full_image_path,
                "target": target,
                "source": data_generator
            }

            if target == 0 and len(test_images_class_0) < 300:
                test_images_class_0.append(image_data)
            elif target == 1 and len(test_images_class_1) < 200:
                test_images_class_1.append(image_data)
            else:
                train_images.append(image_data)

        random.shuffle(test_images_class_0)
        random.shuffle(test_images_class_1)

        test_images_class_0 = test_images_class_0[:300]
        test_images_class_1 = test_images_class_1[:200]

        data["test"].extend(test_images_class_0)
        data["test"].extend(test_images_class_1)
        data["train"].extend(train_images)

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and create JSON metadata file.")
    parser.add_argument("-d", "--data_path", type=str, default="/home/jovyan/nkiselev/ddorin/project/machine_images/data/ArtiFact", help="Path to the dataset directory.")
    parser.add_argument("-o", "--output_file", type=str, default="/home/jovyan/nkiselev/ddorin/project/machine_images/data/dataset.json", help="Output JSON file name.")

    args = parser.parse_args()

    process_dataset(args.data_path, args.output_file)