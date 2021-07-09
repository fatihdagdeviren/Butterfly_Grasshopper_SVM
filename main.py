import numpy as np
import cv2
import imutils
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
from sklearn.externals import joblib
import json
from skimage.feature import hog
from skimage import data, exposure
import xlsxwriter #Yuklemen gerekebilir.

datas = []
labels = []

batch_size = 128


class SVMPredictor():
    def __init__(self, _batchSize = 128):
        self.datas = []
        self.labels = []
        self.batch_size = _batchSize
        self.clf = None
        self.svmResult = None

    def pickleOlustur(self, fileName,object,method=None):
        try:
            if method is None:
                with open(fileName, 'w') as fp:
                    json.dump(object, fp)
            elif method == 1:
                    joblib.dump(object, fileName)
            return '0'
        except BaseException as e:
            print(str(e))
            return '-1'

    def pickleYukle(self, fileName,method=None):
        #data =joblib.load(fileName)
        if method is None:
            with open(fileName, 'r') as fp:
                data = json.loads(fp.read())
        elif method == 1:
            data =joblib.load(fileName)
        self.clf = data

    def createExcelFile(self):
        workbookNames = ["Butterfly-Grasshopper"]
        # Create a workbook and add a worksheet.
        excelPath = "SVMResults.xlsx";
        workbook = xlsxwriter.Workbook(excelPath)
        for wb in workbookNames:
            worksheet = workbook.add_worksheet()
            expenses = [("Data","Prediction","Image Name")]
            # result =  sorted(result, key=lambda x: (x[0]) )
            [expenses.append((x[0],x[1],x[2])) for x in self.svmResult]
            # Start from the first cell. Rows and columns are zero indexed.
            row = 0
            col = 0
            # Iterate over the data and write it out row by row.
            for value in expenses:
                data, tahmin, imageName = value
                worksheet.write(row, col, data)
                worksheet.write(row, col + 1, tahmin)
                worksheet.write(row, col + 2, imageName)
                row += 1
            dogruTahminSayisi = len([x for x in svmTestData if x[0] == x[1]])
            yanlisTahminSayisi = len(svmTestData) - dogruTahminSayisi
            worksheet.write(row, col, "Dogru Tahmin Sayisi")
            worksheet.write(row, col + 1, dogruTahminSayisi)
            worksheet.write(row, col + 2, "Yanlis Tahmin Sayisi")
            worksheet.write(row, col + 3, yanlisTahminSayisi)
            worksheet.write(row, col + 4, "Toplam Sayı")
            worksheet.write(row, col + 5, len(svmTestData))
            worksheet.write(row+1, col, "Dogru Tahmin Orani")
            worksheet.write(row+1, col + 1, dogruTahminSayisi*100/len(svmTestData) )
            worksheet.write(row+1, col + 2, "Yanlis Tahmin Orani")
            worksheet.write(row+1, col + 3, yanlisTahminSayisi*100/len(svmTestData))
            workbook.close()

    def OpencvCanny(self, path, method = None, saveImage = False):
        # load the query image, compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = cv2.imread(path)
        # image = cv2.resize(image)
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        kernel = np.ones((5, 5), np.uint8)
        if method is None:
            # Finding Contours
            # Use a copy of the image e.g. edged.copy()
            # since findContours alters the image
            contours = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(contours)
            if len(cnts) >= 2:
                boxes = []
                for c in cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    # edged[y:y+h,x:x+w]= 0
                    # cv2.imshow('edged', edged)
                    boxes.append([x, y, x + w, y + h])
                    # cv2.imshow('edged1', edged)
                    # cv2.waitKey(0)
                boxes = np.asarray(boxes)
                left = np.min(boxes[:, 0])
                top = np.min(boxes[:, 1])
                right = np.max(boxes[:, 2])
                bottom = np.max(boxes[:, 3])
                edgedCropped = edged[top:bottom, left:right]
                cv2.rectangle(gray, (left, top), (right, bottom), (255, 0, 0), 2)
            else:
                return None
            # print("Number of Contours found = " + str(len(cnts)))
            opening = cv2.morphologyEx(edgedCropped, cv2.MORPH_CLOSE, kernel)
            resultImage = cv2.resize(opening, (200,200))
        elif method  == 1:
            opening = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            if saveImage:
                cv2.imwrite("TestImageResults\\Morph_Open_{0}".format(path.split("\\")[2]),opening )
            openingResized = cv2.resize(opening, (200, 200))
            fd, roi_hog_fd = hog(openingResized, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
            hog_image_rescaled = exposure.rescale_intensity(roi_hog_fd, in_range=(0, 10))
            if saveImage:
                hog_image_Uint8 = hog_image_rescaled / hog_image_rescaled.max()  # normalizes data in range 0 - 255
                hog_image_Uint8 = 255 * hog_image_Uint8
                img = hog_image_Uint8.astype(np.uint8)
                cv2.imwrite("TestImageResults\\Hog_Image.{0}".format(path.split("\\")[2]), img )
            resultImage = hog_image_rescaled
            # cv2.imshow("roi_hog_fd",roi_hog_fd)
            # cv2.imshow("hog_image_rescaled", hog_image_rescaled)
            # cv2.waitKey(0)
        return resultImage
        # cv2.imshow('ContoursCro', edgedCropped)
        # cv2.imshow('opening', opening)
        # cv2.imshow('openingResized', openingResized)
        # cv2.waitKey(0)

    def modelOlustur(self, method=None):
        dosyaYolu = "D:\\trainImages\\train"
        imageList = os.listdir(dosyaYolu)
        for imagePath in imageList:
            print(imagePath)
            path = "{0}\\{1}".format(dosyaYolu, imagePath)
            image = self.OpencvCanny(path,method)
            if image is None:
                continue
            datas.append(np.ravel(np.array(image)))
            if imagePath.__contains__("butterfly"):
                labels.append(0)
            else:
                labels.append(1)
        print("SVM fit basliyor")
        self.clf = svm.SVC()
        self.clf.fit(datas, labels)
        # cv2.imshow("image", image)
        #  cv2.waitKey(0)
         # dosyaya kaydederiz
        # return clf


# def training_step (iterations):
#     start = 0
#     for i in range (iterations):
#         x_batch = datas[start:start+batch_size]
#         y_batch = labels[start:start+batch_size]
#         feed_dict_train = {x: x_batch, y_true: y_batch}
#         sess.run(optimize, feed_dict=feed_dict_train)
#         start += batch_size
#
# def tst_accuracy ():
#     feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
#     acc = sess.run(accuracy, feed_dict=feed_dict_test)
#     print('Testing accuracy:', acc)



if __name__ == "__main__":
    mySvmPredictor = SVMPredictor()
    reset = False
    if reset:
        mySvmPredictor.modelOlustur(1)
        mySvmPredictor.pickleOlustur("svmModelim.pkl",mySvmPredictor.clf,1)
    else:
        mySvmPredictor.pickleYukle("svmModelim.pkl",1)
    svmTestData = []
    dosyaYolu = "D:\\testImages"
    imagePathList = os.listdir(dosyaYolu)
    for path in imagePathList:
        try:
            print("Predict :{0}".format(path))
            fullPath = "{0}\\{1}".format(dosyaYolu, path)
            image = mySvmPredictor.OpencvCanny(fullPath,1,saveImage=True)
            ravelImage = np.ravel(image)
            result = mySvmPredictor.clf.predict([ravelImage])[0]
            if path.__contains__("butterfly"):
                svmTestData.append([0,result, path])
            else:
                svmTestData.append([1, result, path])
        except BaseException as e:
            print("{0} / {1}".format(path,str(e)))

    mySvmPredictor.svmResult = svmTestData
    mySvmPredictor.createExcelFile()





    # x = tf.placeholder(tf.float32, [None, 40000])
    # y_true = tf.placeholder(tf.float32, [None, 2])
    #
    # w = tf.Variable(tf.zeros([40000, 2]))
    # b = tf.Variable(tf.zeros([2]))
    #
    # logits = tf.matmul(x, w) + b
    # y = tf.nn.softmax(logits)
    #
    # xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    # loss = tf.reduce_mean(xent)
    #
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    #
    # training_step(20)

    # tst_accuracy()


