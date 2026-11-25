import pickle
import json
from pathlib import Path

root = (
    Path(__file__).resolve().parents[2]
    / "ObjectConceptLearning"
    / "data"
    / "resources"
)

# for path in root.iterdir():
#     print(path.name)
    
    
with open(root / "OCL_annot_test.pkl", "rb") as f:
    test_data = pickle.load(f)

# print(type(test_data))
# print(len(test_data))
# print(test_data[0])

with open(root / "OCL_class_object.json", "r", encoding="utf-8") as f:
    categories = json.load(f)

# print(type(categories))
# print(len(categories))
# print(categories[0])

with open(root / "OCL_class_attribute.json", "r", encoding="utf-8") as f:
    attribute = json.load(f)
    
with open(root / "OCL_class_affordance.json", "r", encoding="utf-8") as f:
    affordance = json.load(f)
    
# print(type(attribute))
# print(type(affordance))

print(attribute[0])
print(affordance[0])


