import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.utils.random as four

# # A name for the dataset
# name = "fruits"

# # The directory containing the dataset to import
# dataset_dir = "/home/fei/Documents/repo/hackathon/dataset/"

# # The type of the dataset being imported
# dataset_type = fo.types.ImageClassificationDirectoryTree  # for example

# dataset = fo.Dataset.from_dir(
#     dataset_dir=dataset_dir,
#     dataset_type=dataset_type,
#     name=name,
# )

# 1. Load your dataset
dataset = fo.load_dataset("fruits")

four.random_split(
    dataset, 
    {"train": 0.7, "test": 0.2, "val": 0.1}
)

train_view = dataset.match_tags("train")
test_view = dataset.match_tags("test")
val_view = dataset.match_tags("val")

# Check how many samples were tagged for each split
print("Train samples:", dataset.match_tags("train").count())
print("Val samples:  ", dataset.match_tags("val").count())
print("Test samples: ", dataset.match_tags("test").count())

# 6. Export the train split as an ImageClassificationDirectoryTree
#    (one subfolder per class, as determined by `label_field`)
# train_view.export(
#     export_dir="/home/fei/Documents/repo/hackathon/export/train",
#     dataset_type=fo.types.ImageClassificationDirectoryTree,
#     label_field="ground_truth",
# )

# val_view.export(
#     export_dir="/home/fei/Documents/repo/hackathon/export/val",
#     dataset_type=fo.types.ImageClassificationDirectoryTree,
#     label_field="ground_truth",
# )

# test_view.export(
#     export_dir="/home/fei/Documents/repo/hackathon/export/test",
#     dataset_type=fo.types.ImageClassificationDirectoryTree,
#     label_field="ground_truth",
# )

# 7. (Optional) Do the same for val/test if needed
# val_view.export(...)
# test_view.export(...)

# 8. Launch the FiftyOne App to see your data
session = fo.launch_app(dataset)
session.wait()

