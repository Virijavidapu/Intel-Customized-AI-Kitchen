import ultralytics
from ultralytics import YOLO
model = YOLO('D:\Intel_Project\yolosaved_bestdataset_2_epoch25.pt')
results = model('D:\Intel_Project\images1.jpeg')

vegetable_names = []

# Extract the class names from the results and store them in the list
for result in results:
    for class_id in result.boxes.cls:
        class_name = model.names[int(class_id)]  # Convert class_id to integer and get the class name
        vegetable_names.append(class_name)

# Remove duplicates to get unique vegetable names
unique_vegetable_names = list(set(vegetable_names))

# Print the list of vegetable names
print("Predicted Vegetables:", unique_vegetable_names)