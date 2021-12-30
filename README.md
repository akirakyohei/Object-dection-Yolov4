__1. Requirements:

$ pip3 install -r requirements.txt

__2. Convert the darknet Yolo model to tensortflow model:

$ python convert.py -c config_path -w weights_path -o output_path

__3. Run yolo_deep_sort:

$ python main.py -i video_path -c name_class