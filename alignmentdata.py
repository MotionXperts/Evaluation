import pickle

file_path = "/home/c1l1mo/projects/VideoAlignment/result/Axel_merged_one_jump_128/Axel_test_True.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

target_name = "471703066060784233"

for item in data:
    if target_name in str(item.get('name', '')):
        print(f"Matched item: {item}")
        print("Keys in the item:")
        for key in item.keys():
            print(f" - {key}")
