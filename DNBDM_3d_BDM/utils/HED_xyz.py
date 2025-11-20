import cv2 as cv
import numpy as np
import os

# 自定义裁剪层（CropLayer），用于 HED 网络结构中的 "Crop" 操作
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

# 注册自定义层
cv.dnn_registerLayer('Crop', CropLayer)

# 模型路径
prototxt_path = '/hy-tmp/hed-master/examples/hed/deploy.prototxt'
caffemodel_path = '/hy-tmp/hed-master/examples/hed/hed_pretrained_bsds.caffemodel'

# 输入输出路径
input_image_path = '/hy-tmp/P001-corrsta-36size_18slice.png'
output_image_path = '/hy-tmp/edge_output_photo.png'
xyz_output_path = '/hy-tmp/HED_edge_value.csv'  # 可改为 .csv 也行

# 固定的 z 值（第18张切片）
z_value = 17

# 读取模型
net = cv.dnn.readNet(prototxt_path, caffemodel_path)

# 读取图像
frame = cv.imread(input_image_path)
if frame is None:
    raise FileNotFoundError(f"Image not found at {input_image_path}")

# 预处理
inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(500, 500),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
net.setInput(inp)

# 推理
out = net.forward()
out = out[0, 0]
out = cv.resize(out, (frame.shape[1], frame.shape[0]))

# 归一化输出图像并保存
edge_output = (255 * out).astype("uint8")
cv.imwrite(output_image_path, edge_output)
print(f"[INFO] Edge map saved to: {output_image_path}")

# 阈值化处理（例如大于 50 认为是边缘）
threshold = 20
binary_edge = (edge_output > threshold).astype("uint8")

# 获取所有非零像素的坐标（格式为 [ [y1, x1], [y2, x2], ... ]）
edge_coords = np.argwhere(binary_edge > 0)

# 构建 (x, y, z) 坐标列表
xyz_coords = [(int(x), int(y), z_value) for y, x in edge_coords]

# 保存为 CSV 文件（添加表头）
with open(xyz_output_path, 'w') as f:
    f.write("x,y,z\n")  # 写入表头
    for x, y, z in xyz_coords:
        f.write(f"{x},{y},{z}\n")


print(f"[INFO] Edge coordinates saved to: {xyz_output_path}")
