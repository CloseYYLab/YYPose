from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 创建一张空白图像
width, height = 500, 300
#image = Image.new('RGB', (width, height), color='white')
image = Image.open('/home/jvm/HRNet/person.png').convert('RGB')

# 创建一个画笔对象
draw = ImageDraw.Draw(image)

# 定义线条的坐标点
points = [(10, 10), (50, 10), (50, 50), (10, 50), (10, 10)]

# 绘制连线
draw.line(points, fill=(255, 0, 0), width=5)

# 显示图像
plt.imshow(image)
plt.show()
image.save("gaga.jpg")