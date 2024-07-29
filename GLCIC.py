import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from random import randint
import cv2
import os
from imutils import paths

tf.disable_eager_execution()

batchSize = 6
imageHeight = 256
imageWidth = 256
imageChannels = 3
epochs = 50
pp = 70000 # количество изображений, которые будут использоваться для обучения/тестирования
pathM = 'data/masks'
pathMT = 'data/masks_test'
pathI = 'data/faces'


# -------------------------------------------------------------------
# класс, создающий контекстный кодировщик и обучающий его
class GLCIC:
    
    # обрезаем случайную часть изображения для локального дискриминатора
    def randomCrop(self, imageTrue, imageFalse):
        

        imageShape = imageTrue.get_shape()
        boxesTrue = []
        boxesFalse = []
        
        for i in range(imageShape[0]):
            
            x1 = tf.random.uniform(shape=[], minval=0, maxval=imageShape[2]-self.cropWidth, dtype=tf.int32)
            y1 = tf.random.uniform(shape=[], minval=0, maxval=imageShape[1]-self.cropHeight, dtype=tf.int32)
            x2 = x1 + self.cropHeight
            y2 = y1 + self.cropWidth
        
            boxesTrue.append([y1 / imageShape[1], x1 / imageShape[2], y2 / imageShape[1], x2 / imageShape[2]])
            boxesFalse.append([y1 / imageShape[1], x1 / imageShape[2], y2 / imageShape[1], x2 / imageShape[2]])
        
        box_indices = [i for i in range(imageShape[0])]

        newImageTrue = tf.image.crop_and_resize(image = imageTrue, boxes = boxesTrue,box_indices = box_indices, crop_size = [self.cropHeight,self.cropWidth])
        newImageFalse = tf.image.crop_and_resize(image = imageFalse, boxes = boxesFalse,box_indices = box_indices, crop_size = [self.cropHeight,self.cropWidth])
        
        return newImageTrue, newImageFalse
    
    def __init__(self, trainData, testData, epochs, batchSize, imageHeight = 256, imageWidth = 256, imageChannels = 3):
        
        # ----- гиперпараметры обучения
        
        self.epochs = epochs                   # количество эпох
        self.batchSize = batchSize             # размер одного батча
        self.imageHeight = imageHeight         # высота изображения
        self.imageWidth = imageWidth           # ширина изображения
        self.imageChannels = imageChannels     # количество каналов в изображении

        self.reducedHeight = 256     # высота восстанавливаемой области
        self.reducedWidth = 256      # ширина восстанавливаемой области
        self.cropHeight = 128     # высота восстанавливаемой области
        self.cropWidth = 128      # ширина восстанавливаемой области
        
        # ----- объекты и данные, используемые при обучении
        
        # данные для обучения
        self.trainData = trainData
        self.testData = testData
        
        # исходное изображение, исходное изображение с повреждённой областью и маски
        self.inputs =  tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        self.inputsDamaged =  tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        self.masks = tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        
        # генератор
        generator = gen("generator")
        
        # дискриминатор
        discriminator = disConcat("allDiscriminators")
        
        self.Fake = generator(self.inputsDamaged)
        
        # закомментированное нужно, если reducedHeight/reducedWidth отличается от imageHeight/reducedWidth
        # отправляется в дискриминаторы вместо inputs и fake
        #self.reduceImputs = tf.image.resize_images(self.inputs, [self.reducedHeight, self.reducedWidth]) 
        #self.reduceFake = tf.image.resize_images(self.Fake, [self.reducedHeight, self.reducedWidth])
        
        self.cropTrue, self.cropFake = self.randomCrop(self.inputs, self.Fake)
        
        self.disTrue = discriminator(self.inputs, self.cropTrue)
        self.disFake = discriminator(self.Fake, self.cropFake)
        
        self.disLoss = -tf.reduce_mean(tf.log(self.disTrue + 1e-6) + tf.log(1 - self.disFake + 1e-6))
        
        self.genLoss1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.masks*(self.inputs - self.Fake)), [1, 2, 3]))
        self.genLoss2 = (-0.0004)*tf.reduce_mean(tf.log(self.disFake + 1e-5)) + tf.reduce_mean(tf.reduce_sum(tf.square(self.inputs - self.Fake), [1, 2, 3]))
        
        self.disOptimizer = tf.train.AdamOptimizer(2e-4).minimize(self.disLoss, var_list=discriminator.get_var())
        self.genOptimizer1 = tf.train.AdamOptimizer(2e-4).minimize(self.genLoss1, var_list=generator.get_var())
        self.genOptimizer2 = tf.train.AdamOptimizer(2e-4).minimize(self.genLoss2, var_list=generator.get_var())
        
        self.costDis = tf.summary.scalar("disLoss", self.disLoss)
        self.costGen = tf.summary.scalar("genLoss", self.genLoss2)
        self.merged = tf.summary.merge_all()
        self.writerTest = tf.summary.FileWriter("./logs/test")
        self.writerTrain = tf.summary.FileWriter("./logs/train")
        
        self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        
    # -------------------------------------------------------------- обучение
    def train(self, i=0, whenSave = 1):
        
        tf.reset_default_graph()
        
        self.writerTrain.add_graph(self.sess.graph)
        self.writerTest.add_graph(self.sess.graph)
        
        for epoch in range(self.epochs):
            
            im = []
            im2 = []
            
            # ----- шаг обучения
            for numberBatch in range(len(self.trainData)):
                
                im, damagedIm, mask = self.trainData[numberBatch]
                
                #self.sess.run(self.genOptimizer1, feed_dict={self.inputs: im, self.inputsDamaged: damagedIm, self.masks: mask})
                self.sess.run(self.disOptimizer, feed_dict={self.inputs: im, self.inputsDamaged: damagedIm})
                self.sess.run(self.genOptimizer2, feed_dict={self.inputs: im, self.inputsDamaged: damagedIm, self.masks: mask})
            
            # ----- вывод промежуточных результатов:
            im2, damagedIm2, mask2 = self.testData[0]

            summaryTrain, resLossDTrain, resLossGTrain = self.sess.run([self.merged, self.disLoss, self.genLoss2], feed_dict={self.inputs: im, self.inputsDamaged: damagedIm, self.masks: mask})

            summaryTest, resLossDTest, resLossGTest = self.sess.run([self.merged, self.disLoss, self.genLoss2], feed_dict={self.inputs: im2, self.inputsDamaged: damagedIm2, self.masks: mask2})

            self.writerTrain.add_summary(summaryTrain, i)
            self.writerTest.add_summary(summaryTest, i)

            print("Итерация " + str(i) + ". D loss = " + str(resLossDTrain) + ", D loss Test = " + str(resLossDTest) + ", G loss = " + str(resLossGTrain) + ", G loss Test = " + str(resLossGTest) + ".")

            # ----- сохраняем параметры нейронки каждые whenSave эпох
            if (epoch + 1) % 1 == 0:

                resImage = self.sess.run(self.Fake, feed_dict={self.inputsDamaged: damagedIm2})
                Image.fromarray(np.uint8(resImage[0]*255)).save("./Results//" + str(i) + ".jpg")
                Image.fromarray(np.uint8(damagedIm2[0]*255)).save("./Results//" + str(i) + "_1.jpg")
                self.saver.save(self.sess, "./save_para//para.ckpt")

            self.trainData.on_epoch_end()
            self.testData.on_epoch_end()
            i = i+1
            
    # -------------------------------------------------------------- восстановление данных
    def restoreModel(self, pathMeta, path):

        self.saver = tf.train.import_meta_graph(pathMeta)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
    
    
    # -------------------------------------------------------------- 
    # использование готовой модели для восстановления изображения:
    def useModel(self, image):
        
        resImage = self.sess.run([self.outputs], feed_dict={self.inputsDamaged: image})
        Image.fromarray(np.uint8(resImage[0][0]*255)).save("./Results.jpg")
        print("Результат сохранен")
    
    
# -------------------------------------------------------------------
# реализация слоёв нейронной сети 

# ----- свёртка
def conv2D(name, inputs, filters, kSize, stride, padding):
    
    with tf.variable_scope(name):
        
        W = tf.get_variable("W", shape=[kSize, kSize, inputs.shape[-1], filters], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        
        return  tf.math.add(tf.nn.conv2d(inputs, W, [1, stride, stride, 1], padding), b)

    
# ----- расширенная свёртка
def dilatedConv2D(name, inputs, filters, kSize, dilation):
    
    with tf.variable_scope(name):
        
        W = tf.get_variable("W", shape=[kSize, kSize, inputs.shape[-1], filters], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        
        return  tf.math.add(tf.nn.atrous_conv2d(inputs, W, dilation, padding='SAME'), b)

# ----- обратная свёртка
def unconv2D(name, inputs, filters, kSize, stride, padding, r = 2):
    
    with tf.variable_scope(name):
        
        w = tf.get_variable("W", shape=[kSize, kSize, filters, int(inputs.shape[-1])], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        
        B = tf.shape(inputs)[0]
        H = int(inputs.shape[1])
        W = int(inputs.shape[2])
        
        return  tf.math.add(tf.nn.conv2d_transpose(inputs, w, [B, H*r, W*r, filters], [1, stride, stride, 1], padding), b)

# ----- полносвязный слой / вектор 
def fullyConnected(name, inputs, filters):
    
    inputs = tf.layers.flatten(inputs)
    
    with tf.variable_scope(name):
        
        W = tf.get_variable("W", [int(inputs.shape[-1]), filters], initializer=tf.random_normal_initializer(0., 0.005))
        b = tf.get_variable("b", [filters], initializer=tf.constant_initializer(0.))
    
        return tf.math.add(tf.matmul(inputs, W), b)


# -------------------------------------------------------------------
# класс генератора
class gen:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv1", inputs, 64, 5, 1, "SAME")))
            
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv2", model, 128, 3, 2, "SAME")))
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv3", model, 128, 3, 1, "SAME")))
            
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv4", model, 256, 3, 2, "SAME")))
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv5", model, 256, 3, 1, "SAME")))
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv6", model, 256, 3, 1, "SAME")))
            
            model = tf.nn.relu(tf.layers.batch_normalization(dilatedConv2D("dilConv1", model, 256, 3, 2)))
            model = tf.nn.relu(tf.layers.batch_normalization(dilatedConv2D("dilConv2", model, 256, 3, 4)))
            model = tf.nn.relu(tf.layers.batch_normalization(dilatedConv2D("dilConv3", model, 256, 3, 8)))
            model = tf.nn.relu(tf.layers.batch_normalization(dilatedConv2D("dilConv4", model, 256, 3, 16)))
            
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv7", model, 256, 3, 1, "SAME")))
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv8", model, 256, 3, 1, "SAME")))
            
            model = tf.nn.relu(tf.layers.batch_normalization(unconv2D("unconv1", model, 128, 4, 2, "SAME")))
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv9", model, 128, 3, 1, "SAME")))
            
            model = tf.nn.relu(tf.layers.batch_normalization(unconv2D("unconv2", model, 64, 4, 2, "SAME")))
            model = tf.nn.relu(tf.layers.batch_normalization(conv2D("conv10", model, 32, 3, 1, "SAME")))
            model = tf.nn.tanh(conv2D("conv11", model, 3, 3, 1, "SAME"))
    
            return model

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

# -------------------------------------------------------------------
# класс локального дискриминатора
class disLocal:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv1", inputs, 64, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv2", model, 128, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv3", model, 256, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv4", model, 512, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv5", model, 512, 5, 2, "SAME")))
            model = fullyConnected("fc", model, 1024)
            
            return model

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    
# -------------------------------------------------------------------
# класс глобального дискриминатора
class disGlobal:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv1", inputs, 64, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv2", model, 128, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv3", model, 256, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv4", model, 512, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv5", model, 512, 5, 2, "SAME")))
            model = tf.layers.batch_normalization(tf.nn.relu(conv2D("conv6", model, 512, 5, 2, "SAME")))
            model = fullyConnected("fc", model, 1024)
    
            return model

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

# -------------------------------------------------------------------
# объединение дискриминаторов
class disConcat():
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputsG, inputsL):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            globalDis = disGlobal('global')
            localDis = disLocal('local')
            
            output = tf.concat((globalDis(inputsG), localDis(inputsL)), 1)
            output = tf.sigmoid(fullyConnected("fc", output, 1))
    
            return output

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


# -------------------------------------------------------------------
# класс, генерирующий тренировочные данные
class createAugment():
    
    # --
    # инициализация объекта класса
    def __init__(self, imgs, masks, batch_size=10, dim=(128, 128), n_channels=3):
        self.batch_size = batch_size  # размер батча
        self.images = imgs            # исходное изображение
        self.masks = masks            # маски изображений
        self.dim = dim                # размер изображения
        self.n_channels = n_channels  # количество каналов
        self.on_epoch_end()           # генерация набора батчей
    
    # --
    # результат: кол-во возможных батчей за эпоху
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    # --
    # результат: взятие батча с заданным номером (индексом)
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        imageOrig, imageMasked, imageMasks = self.data_generation(indexes)
        return imageOrig, imageMasked, imageMasks
    
    # --
    # функция, повторяющаяся в конце каждой эпохи
    # результат: новая совокупность индексов изображений для очередного батча
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        np.random.shuffle(self.indexes)
    
    # --
    # результат: батч данных, включающий в себя 
    # исходное изображени, маскированное изображение и маску
    def data_generation(self, idxs):
        
        imageMasked = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # маскированное изображения
        imageMasks = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # маски
        imageOrig = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # изображение под маской

        for i, idx in enumerate(idxs):
            
            image, masked_image, image_masks = self.createMask(self.images[idx].copy())
            imageMasked[i,] = masked_image/255
            imageOrig[i,] = image/255
            imageMasks[i,] = image_masks/255
            
        return imageOrig, imageMasked, imageMasks
    
    # --
    # поворот изображения
    def imageRotate(self, img):
        
        angle = np.random.randint(1, 359)
        M = cv2.getRotationMatrix2D((self.dim[0]//2, self.dim[1]//2), 45, 1.0)
        rotated = cv2.warpAffine(img, M, self.dim)
        
        return rotated
    
    # --
    # уменьшение изначальной маски
    def imageResize(self, img):

        background = np.full((self.dim[0], self.dim[1], self.n_channels), 0, np.uint8)

        widthNew = np.random.randint(int(self.dim[0]/2), self.dim[0])
        heightNew = np.random.randint(int(self.dim[1]/2), self.dim[1])
        
        imgNew = cv2.resize(img, (widthNew, heightNew))

        xNew = np.random.randint(0, self.dim[0] - widthNew)
        yNew = np.random.randint(0, self.dim[1] - heightNew)

        background[yNew:yNew+heightNew,xNew:xNew+widthNew] = imgNew

        return background
    
    # --
    # добавляем дополнительные элементы
    def imageDetails(self, mask):
        
        # генерируем количество линий-повреждений на рисунке
        n_line = np.random.randint(1, 5)
        
        # рисуем линии
        for i in range(n_line):
            
            # генерируем первую точку линии
            x_start = np.random.randint(1, self.dim[0])
            y_start = np.random.randint(1, self.dim[1])
            
            # генерируем вторую точку линии
            x_finish = np.random.randint(1, self.dim[0])
            y_finish = np.random.randint(1, self.dim[1])
            
            # определяем толщину линии
            point = np.random.randint(1, 5)
            
            # рисуем линию между сгенерированными точками
            cv2.line(mask, (x_start, y_start), (x_finish, y_finish), (255,255,255), point)
        
        return mask
    
    # --
    # маскируем входное изображение случайной маской повреждений
    def createMask(self, image):
        
        randNumberOfMask = np.random.randint(0, len(self.masks)-1)
        isRotate = np.random.randint(1, 10)
        isResize = np.random.randint(1, 10)
        
        # маска, значение которой 255 в области, которую нужно заполнить
        # 0 - в остальных областях
        mask = (self.masks[randNumberOfMask].copy())
        
        if isResize%2 == 0:
            mask = self.imageResize(mask)
        
        if isRotate%2 == 0:
            mask = self.imageRotate(mask)
          
        
        mask = self.imageDetails(mask)
        
        # делаем края маски более жесткими (чтобы пиксели маски не содержали значения, кроме 0 и 255)
        mask2 = (mask//255)*255
        
        imageMasked = cv2.bitwise_and(image, cv2.bitwise_not(mask2)) + mask2
        
        return image, imageMasked, mask2

imagePaths = os.listdir(pathI) 
masksPaths = os.listdir(pathM) 
masksPathsTest = os.listdir(pathMT)
images = np.empty((pp, imageHeight, imageWidth, imageChannels), dtype='uint8')
masks = np.empty((len(masksPaths), imageHeight, imageWidth, imageChannels), dtype='uint8')
masksTest = np.empty((len(masksPathsTest), imageHeight, imageWidth, imageChannels), dtype='uint8')

i = 0

for path in imagePaths:
    img = Image.open(os.path.join(pathI, path))
    img = img.resize((imageHeight,imageWidth))
    images[i] = tf.keras.preprocessing.image.img_to_array(img)
    i = i+1
    if i == pp:
        break

i = 0
for path in masksPaths:
    img = Image.open(os.path.join(pathM, path))
    img = img.resize((imageHeight,imageWidth))
    masks[i] = tf.keras.preprocessing.image.img_to_array(img)
    i = i+1

i = 0    
for path in masksPathsTest:
    img = Image.open(os.path.join(pathMT, path))
    img = img.resize((imageHeight,imageWidth))
    masksTest[i] = tf.keras.preprocessing.image.img_to_array(img)
    i = i+1
    
trainData = createAugment(images[0:int(pp*0.9)], masks, batchSize, dim = [imageHeight, imageWidth])
testData = createAugment(images[int(pp*0.9):], masksTest, batchSize, dim = [imageHeight, imageWidth])

network = GLCIC(trainData, testData, epochs, batchSize, imageHeight, imageWidth, imageChannels)

network.restoreModel('./save_para//para.ckpt.meta', './save_para')

with tf.device('/gpu:0'):
    network.train(152)