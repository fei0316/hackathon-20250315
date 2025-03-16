import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.utils.random as four
from ultralytics import YOLO

# A name for the dataset
name = "fruits"

# # The directory containing the dataset to import
dataset_dir = "/home/fei/Documents/repo/hackathon/dataset/"

# # The type of the dataset being imported
dataset_type = fo.types.ImageClassificationDirectoryTree  # for example

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
    persistent=True
)

# 1. Load your dataset
# dataset = fo.load_dataset("fruits")

four.random_split(
    dataset, 
    {"train": 0.17, "test": 0.03, "val": 0.8}
)

train_view = dataset.match_tags("train")
test_view = dataset.match_tags("test")
val_view = dataset.match_tags("val")

# # Check how many samples were tagged for each split
print("Train samples:", dataset.match_tags("train").count())
print("Test samples: ", dataset.match_tags("test").count())
print("Validation samples: ", dataset.match_tags("val").count())

# 6. Export the train split as an ImageClassificationDirectoryTree
#    (one subfolder per class, as determined by `label_field`)
train_view.export(
    export_dir="/home/fei/Documents/repo/hackathon/datasets/train",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    label_field="ground_truth",
    export_media="symlink",
)

test_view.export(
    export_dir="/home/fei/Documents/repo/hackathon/datasets/test",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    label_field="ground_truth",
    export_media="symlink",
)

val_view.export(
    export_dir="/home/fei/Documents/repo/hackathon/datasets/val",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    label_field="ground_truth",
    export_media="symlink",
)

# The path to the `dataset.yaml` file we created above
# YAML_FILE = "/home/fei/Documents/repo/hackathon/datasets/"

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model
# model = YOLO("yolov8s.yaml")  # build a model from scratch

# Train the model
model.train(data="/home/fei/Documents/repo/hackathon/datasets/", epochs=3,
                    imgsz=320, batch=12, project="fruits-yolo", exist_ok=True)

# Evaluate model on the validation set
metrics = model.val()

# Export the model
path = model.export(format="onnx")



# 8. Launch the FiftyOne App to see your data
# session = fo.launch_app(dataset)
# session.wait()

