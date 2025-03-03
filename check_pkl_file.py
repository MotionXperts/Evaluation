import pickle

file_path = '/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    
for item in data:
    print(item["name"])
    feature = item["features"]
    length = len(feature)
    print(f"Length of features: {length}")
    annotations_label = item['annotations_label']

    # Find all annotations label == 1
    indices = (annotations_label == 1).nonzero(as_tuple=True)[0]

    # Get start and end index
    if len(indices) > 0:
        start_idx = indices[0].item()
        end_idx = indices[-1].item()
        print(f"Start index: {start_idx}, End index: {end_idx}")
    if ((int(end_idx)) -  (int(start_idx)) != length ):
        print("error")

print(type(data))
print(data)
