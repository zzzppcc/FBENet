import os
import cv2
datasets = ["DUTS"]
for dataset in datasets:
    print(dataset)

    input_path = os.path.join("data",dataset,"mask")
    output_path = os.path.join("data",dataset,"edge")
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    for file in os.listdir(input_path):
        read_path = os.path.join(input_path,file)
        mask = cv2.imread(read_path,cv2.IMREAD_GRAYSCALE)
        mask = cv2.Canny(mask,128,256)
        write_path = os.path.join(output_path,file)
        cv2.imwrite(write_path,mask)

