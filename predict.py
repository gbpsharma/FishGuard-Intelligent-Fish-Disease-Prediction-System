import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from model_definition import FishDiseaseModel


device = torch.device("cpu")

class_names = [
"Bacterial Red disease",
"Bacterial diseases - Aeromoniasis",
"Bacterial gill disease",
"Fungal diseases Saprolegniasis",
"Healthy Fish",
"Parasitic diseases",
"Viral diseases White tail disease"
]

model = FishDiseaseModel(len(class_names))
model.load_state_dict(torch.load("fish_disease_model.pth",map_location=device))
model.eval()

transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],
	                     [0.229,0.224,0.225])
])

Tk().withdraw()
image_path = askopenfilename(title="Select Fish Image")

print("\nSelected:",image_path)

image = Image.open(image_path).convert("RGB")
original = np.array(image)

# =========================
# GRADCAM HOOKS
# =========================
activations = None
gradients = None

def forward_hook(module,input,output):
	global activations
	activations = output

def backward_hook(module,grad_input,grad_output):
	global gradients
	gradients = grad_output[0]

target_layer = model.backbone.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# PREDICTION
# =========================
input_tensor = transform(image).unsqueeze(0)

output = model(input_tensor)

probabilities = F.softmax(output,dim=1)
confidence,predicted = torch.max(probabilities,1)

predicted_class = class_names[predicted.item()]
confidence_score = confidence.item()*100

print("\nPrediction:",predicted_class)
print("Confidence:",round(confidence_score,2),"%")

# =========================
# CONFIDENCE FILTER
# =========================
CONF_THRESHOLD = 80.0

if confidence_score < CONF_THRESHOLD:
	print("\n❌ No Fish Detected")

	plt.imshow(original)
	plt.title("No Fish Detected")
	plt.axis("off")
	plt.show()

	exit()

# =========================
# ✅ HEALTHY CHECK (NEW)
# =========================
if predicted_class == "Healthy Fish":
	print("\n✔ Healthy Fish - No disease region")

	plt.imshow(original)
	plt.title("Healthy Fish\n" + str(round(confidence_score,2)) + "%")
	plt.axis("off")
	plt.show()

	exit()

# =========================
# GRADCAM
# =========================
model.zero_grad()
output[0][predicted].backward()

weights = torch.mean(gradients,dim=(2,3),keepdim=True)

cam = torch.sum(weights*activations,dim=1).squeeze()
cam = torch.relu(cam).detach().numpy()

cam = cv2.resize(cam,(224,224))
cam = (cam-cam.min())/(cam.max()-cam.min())

heatmap = np.uint8(255*cam)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

original_resized = cv2.resize(original,(224,224))
overlay = heatmap*0.4 + original_resized*0.6
overlay = overlay.astype(np.uint8)

# =========================
# ROBUST BOUNDING BOX
# =========================
gray = cv2.cvtColor(heatmap,cv2.COLOR_RGB2GRAY)
norm = gray / np.max(gray)

kernel = np.ones((5,5),np.uint8)
best_cnt = None

# STRICT
thresh = np.where(norm > 0.8,255,0).astype(np.uint8)
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
	best_cnt = max(contours,key=cv2.contourArea)

# FALLBACK
if best_cnt is None:
	thresh = np.where(norm > 0.6,255,0).astype(np.uint8)
	thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

	contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) > 0:
		best_cnt = max(contours,key=cv2.contourArea)

boxed = overlay.copy()

if best_cnt is not None:
	x,y,w,h = cv2.boundingRect(best_cnt)

	cv2.rectangle(boxed,(x,y),(x+w,y+h),(255,0,0),3)

# =========================
# DISPLAY
# =========================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(overlay)
plt.title("GradCAM")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(boxed)
plt.title(predicted_class + "\n" + str(round(confidence_score,2)) + "%")
plt.axis("off")

plt.show()