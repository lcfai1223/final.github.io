from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PIL import Image

import q1, q5
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class test_dialogue(QWidget):
	def __init__(self, img):
		super().__init__()
		layout = QVBoxLayout()
		self.label = QLabel("Test Image Index")
		self.E1 = QLineEdit()
		self.btn = QPushButton("Inference")

		layout.addWidget(self.label)
		layout.addWidget(self.E1)
		layout.addWidget(self.btn)

		self.setLayout(layout)


class GroupBox(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		self.setWindowTitle("影像處理分析")
		layout = QGridLayout()
		self.setLayout(layout)
		self.setFixedSize(1400, 800)
		
		total_widgets = []
		# total_widgets.append(self.c2("2. Image Smoothing"))
		total_widgets.append(self.c1("Final Project Image Processing"))
		total_widgets.append(self.c2("Final Project Machine Learning"))


		for idx, widget in enumerate(total_widgets):
			layout.addWidget(widget, 0, idx)

	def c1(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)
		
		Img = QLabel(self)

		b1 = QPushButton("Gaussian Blur")
		b2 = QPushButton("Sobel X")
		b3 = QPushButton("Sobel Y")
		b4 = QPushButton("Magnitude")
		img_btn = QPushButton("Open Image")
		down_img_btn = QPushButton("Download Image")
		
		widgets = [b1, b2, b3, b4, img_btn, down_img_btn]
		
		for w in widgets:
			w.setMinimumWidth(100)
			w.setMinimumHeight(50)

		b1.clicked.connect(lambda: q1.gaussian_blur(Img) )
		b2.clicked.connect(lambda: q1.sobel_x(Img) )
		b3.clicked.connect(lambda: q1.sobel_y(Img) )
		b4.clicked.connect(lambda: q1.magnitude(Img) )
		img_btn.clicked.connect(lambda: self.getImage("./Q1_Image/edge.png", Img))
		down_img_btn.clicked.connect(self.downImage)

		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		vbox.addWidget(b4)
		vbox.addWidget(Img)
		vbox.addWidget(img_btn)
		vbox.addWidget(down_img_btn)
		return groupbox	


	def getImage(self, save_file_path, Img):
		fname = QFileDialog.getOpenFileName(self, 'Open file',
											'', "Image files (*.jpeg *.jpg *.gif *.png)")
		imagePath = fname[0]
		if imagePath:
			acc = Image.open(imagePath)
			acc.save(save_file_path)
		
		pixmap = QPixmap(imagePath)
		pixmap = pixmap.scaled(600, 600, QtCore.Qt.KeepAspectRatio)
		Img.setPixmap(QPixmap(pixmap))

	def downImage(self):
		fname, _ = QFileDialog.getSaveFileName(
			self, "Save audio file", "download.png", "(*.jpeg *.jpg *.gif *.png)"
		)
		if fname:
			print(fname)
			acc = Image.open("./Q1_image/edge.png")
			acc.save(fname)

	def c2(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)
		Img = QLabel(self)
		b1 = QPushButton("1. Show Train Image")
		b2 = QPushButton("2. Show Hyperparameters")
		b3 = QPushButton("3. Show Accuracy")
		b4 = QPushButton("4. Test")
		img_btn = QPushButton("Open Image")

		widgets = [b1, b2, b3, b4, img_btn]
		
		for w in widgets:
			w.setMinimumWidth(100)
			w.setMinimumHeight(50)


		b1.clicked.connect(lambda:q5.show_train_image(Img) )
		b2.clicked.connect(lambda:q5.show_params() )
		b3.clicked.connect(lambda:q5.show_accuracy(Img) )
		b4.clicked.connect(lambda: q5.test_image(Img))
		img_btn.clicked.connect(lambda: self.getImage("./test/test.png", Img))

		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		vbox.addWidget(b4)
		vbox.addWidget(Img)
		vbox.addWidget(img_btn)
		return groupbox
	# def on_pushButton_clicked(self, img):
	# 	self.w = test_dialogue(img)
	# 	self.w.show()


app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())