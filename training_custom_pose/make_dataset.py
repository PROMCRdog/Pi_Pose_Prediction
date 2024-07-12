import cv2
import os

# 定义要拍摄图片的类名
current_class = "test"  # 将此更改为你想要拍摄图片的类
#  ____________________________________________________
# ¦注意如果重复运行此脚本，若不修改类名称将会覆盖以拍摄的图片数据¦
#  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
                        
# 如果目录不存在，则创建目录
base_dir = "./my_data"
class_dir = os.path.join(base_dir, current_class)
os.makedirs(class_dir, exist_ok=True)

# 初始化摄像头
cap = cv2.VideoCapture(2) # 修改成正确摄像头编号


# 已捕获图像数量的计数器
image_counter = 0

def capture_image(event, x, y, flags, param):
    global image_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # 捕获图像
        ret, frame = cap.read()
        if ret:
            # 定义图像保存路径
            image_path = os.path.join(class_dir, f"{current_class}_{image_counter:05d}.jpg")
            # 保存图像
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            image_counter += 1

# 设置窗口和鼠标回调函数
cv2.namedWindow("Capture Images")
cv2.setMouseCallback("Capture Images", capture_image)

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 显示图像窗口
    cv2.imshow("Capture Images", frame)

    # 如果按下 'q' 键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()