import torch
import torch.nn.functional as F
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import cv2
import io
import base64
import sys
import timm

from ultralytics import YOLO

sys.path.append("..")
from model_definition import FishDiseaseModel

# =========================
# INIT
# =========================
app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

device = torch.device("cpu")

# =========================
# FISH VALIDATOR
# =========================
fish_validator = timm.create_model(
	"resnet18",
	pretrained=True
)

fish_validator.eval()

imagenet_transform = ResNet18_Weights.DEFAULT.transforms()

fish_keywords = [
	"fish",
	"shark",
	"ray",
	"eel",
	"goldfish",
	"stingray",
	"puffer",
	"barracouta"
]

# =========================
# YOLO
# =========================
yolo = YOLO("yolov8n.pt")

# =========================
# CLASS NAMES
# =========================
class_names = [
	"Bacterial Red disease",
	"Bacterial diseases - Aeromoniasis",
	"Bacterial gill disease",
	"Fungal diseases Saprolegniasis",
	"Healthy Fish",
	"Parasitic diseases",
	"Viral diseases White tail disease"
]

# =========================
# LOAD DISEASE MODEL
# =========================
model = FishDiseaseModel(len(class_names))

model.load_state_dict(
	torch.load(
		"../fish_disease_model.pth",
		map_location=device
	)
)

model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(
		[0.485,0.456,0.406],
		[0.229,0.224,0.225]
	)
])

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

target_layer = model.backbone.layer4[-1]

target_layer.register_forward_hook(
	forward_hook
)

target_layer.register_full_backward_hook(
	backward_hook
)

# =========================
# VALIDATE FISH IMAGE
# =========================
def is_fish_image(image):

	input_tensor = imagenet_transform(
		Image.fromarray(image)
	).unsqueeze(0)

	with torch.no_grad():

		output = fish_validator(input_tensor)

		probabilities = F.softmax(
			output,
			dim=1
		)

		top5_prob,top5_catid = torch.topk(
			probabilities,
			5
		)

		labels = ResNet18_Weights.DEFAULT.meta["categories"]

		for i in range(5):

			label = labels[
				top5_catid[0][i]
			].lower()

			for keyword in fish_keywords:

				if keyword in label:

					return True

	return False

# =========================
# API
# =========================
@app.post("/predict")
async def predict(file:UploadFile=File(...)):

	try:

		# =========================
		# READ IMAGE
		# =========================
		contents = await file.read()

		image = Image.open(
			io.BytesIO(contents)
		).convert("RGB")

		original = np.array(image)

		# =========================
		# CHECK FISH IMAGE
		# =========================
		if not is_fish_image(original):

			return {
				"predicted_class":"NOT_FISH",
				"confidence":0,
				"image":"",
				"gradcam":"",
				"infected":"",
				"top3":[]
			}

		# =========================
		# YOLO DETECTION
		# =========================
		results = yolo(
			original,
			conf=0.20
		)

		boxes = results[0].boxes.xyxy.cpu().numpy()

		# =========================
		# NO OBJECT
		# =========================
		if len(boxes)==0:

			x1,y1,x2,y2 = (
				0,
				0,
				original.shape[1],
				original.shape[0]
			)

		else:

			best_box = None
			best_area = 0

			for i in range(len(boxes)):

				x1_temp,y1_temp,x2_temp,y2_temp = boxes[i]

				area = (
					(x2_temp-x1_temp) *
					(y2_temp-y1_temp)
				)

				if area > best_area:

					best_area = area
					best_box = boxes[i]

			x1,y1,x2,y2 = map(
				int,
				best_box
			)

		# =========================
		# CROP
		# =========================
		fish_crop = original[
			y1:y2,
			x1:x2
		]

		# =========================
		# MODEL INPUT
		# =========================
		input_tensor = transform(
			Image.fromarray(fish_crop)
		).unsqueeze(0)

		# =========================
		# PREDICTION
		# =========================
		output = model(input_tensor)

		probabilities = F.softmax(
			output,
			dim=1
		)

		confidence,predicted = torch.max(
			probabilities,
			1
		)

		confidence_score = float(
			confidence.item()*100
		)

		predicted_class = class_names[
			predicted.item()
		]

		# =========================
		# TOP 3
		# =========================
		top_probs,top_indices = torch.topk(
			probabilities,
			3
		)

		top3 = []

		for i in range(3):

			top3.append([
				class_names[
					top_indices[0][i].item()
				],
				float(
					top_probs[0][i].item()*100
				)
			])

		# =========================
		# HEALTHY
		# =========================
		if predicted_class == "Healthy Fish":

			_,buffer = cv2.imencode(
				".jpg",
				fish_crop
			)

			img_base64 = base64.b64encode(
				buffer
			).decode()

			return {
				"predicted_class":"Healthy Fish",
				"confidence":confidence_score,
				"image":img_base64,
				"gradcam":"",
				"infected":"",
				"top3":top3
			}

		# =========================
		# GRADCAM
		# =========================
		model.zero_grad()

		output[0][predicted].backward()

		weights = torch.mean(
			gradients,
			dim=(2,3),
			keepdim=True
		)

		cam = torch.sum(
			weights*activations,
			dim=1
		).squeeze()

		cam = torch.relu(cam).detach().numpy()

		cam = cv2.resize(
			cam,
			(
				fish_crop.shape[1],
				fish_crop.shape[0]
			),
			interpolation=cv2.INTER_CUBIC
		)

		cam = cam - np.min(cam)

		if np.max(cam)!=0:
			cam = cam / np.max(cam)

		cam = cv2.GaussianBlur(
			cam,
			(11,11),
			0
		)

		heatmap = np.uint8(255*cam)

		heatmap = cv2.applyColorMap(
			heatmap,
			cv2.COLORMAP_JET
		)

		heatmap = cv2.cvtColor(
			heatmap,
			cv2.COLOR_BGR2RGB
		)

		overlay = cv2.addWeighted(
			fish_crop,
			0.55,
			heatmap,
			0.45,
			0
		)

		# =========================
		# ENCODE IMAGES
		# =========================
		_,buffer1 = cv2.imencode(
			".jpg",
			fish_crop
		)

		original_base64 = base64.b64encode(
			buffer1
		).decode()

		_,buffer2 = cv2.imencode(
			".jpg",
			overlay
		)

		gradcam_base64 = base64.b64encode(
			buffer2
		).decode()

		# =========================
		# INFECTED AREA
		# =========================
		gray = cv2.cvtColor(
			heatmap,
			cv2.COLOR_RGB2GRAY
		)

		norm = gray / 255.0

		kernel = np.ones(
			(5,5),
			np.uint8
		)

		thresh = np.where(
			norm > 0.65,
			255,
			0
		).astype(np.uint8)

		thresh = cv2.morphologyEx(
			thresh,
			cv2.MORPH_CLOSE,
			kernel
		)

		contours,_ = cv2.findContours(
			thresh,
			cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE
		)

		infected = overlay.copy()

		if len(contours)>0:

			cnt = max(
				contours,
				key=cv2.contourArea
			)

			x,y,w,h = cv2.boundingRect(cnt)

			cv2.rectangle(
				infected,
				(x,y),
				(x+w,y+h),
				(255,0,0),
				4
			)

		_,buffer3 = cv2.imencode(
			".jpg",
			infected
		)

		infected_base64 = base64.b64encode(
			buffer3
		).decode()

		# =========================
		# RESPONSE
		# =========================
		return {
			"predicted_class":predicted_class,
			"confidence":confidence_score,
			"image":original_base64,
			"gradcam":gradcam_base64,
			"infected":infected_base64,
			"top3":top3
		}

	except Exception as e:

		return {
			"error":str(e)
		}