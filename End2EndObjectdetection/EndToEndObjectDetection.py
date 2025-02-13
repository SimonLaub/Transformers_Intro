from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

# https://huggingface.co/facebook/detr-resnet-50

image_path = "C:\\Users\\zila\\beijing.jpg"
image = Image.open("C:\\Users\\zila\\beijing.jpg")

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

'''for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )'''

import cv2
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt

# Filter boxes based on confidence score (threshold can be adjusted)
scores = results["scores"].detach().numpy()
keep = scores > 0.9
boxes = results["boxes"].detach().numpy()[keep]
labels = results["labels"].detach().numpy()[keep]
scores = scores[keep]

for box, label, score in zip(boxes, labels, scores):
    xmin, ymin, xmax, ymax = box
    box_width = xmax - xmin
    box_height = ymax - ymin
    img_width, img_height = image_rgb.shape[:2]
    font_scale = 0.3 #calculate_label_size(box_width, box_height, img_width, img_height, max_scale=1)

    label_text = f"{model.config.id2label[label]}: {score:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        1
    )

    # Get the color for this label
    color = (0 , 255, 0) # get_label_color(label)

    # Draw rectangle and label with the same color for the same class
    cv2.rectangle(image_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, max(2, int(font_scale * 3)))
    cv2.rectangle(image_rgb, (int(xmin), int(ymin) - text_height - baseline - 5),
                  (int(xmin) + text_width, int(ymin)), color, -1)
    cv2.putText(image_rgb, label_text, (int(xmin), int(ymin) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(font_scale * 1.5)), cv2.LINE_AA)

    # Step 5: Plot the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("off")  # Hide axes
plt.show()