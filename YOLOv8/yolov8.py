from ultralytics import YOLO


if __name__ == '__main__':
	model = YOLO("yolov8x.pt") 
	results = model("https://ultralytics.com/images/bus.jpg",save=True) 