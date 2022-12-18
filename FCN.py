# FCN

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
from IPython.display import clear_output
import matplotlib.pyplot as plt
from data import dataset, process_path, preprocess

image_ds = dataset.map(process_path)
data = image_ds.map(preprocess)

TRAIN_LENGTH = data.splits['train'].num_examples
BUFFER_SIZE = 1000
BATCH_SIZE = 128
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = data['train']
test = data['test']
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])


def fcn(image_size, ch_in=3, ch_out=3):
    inputs = Input(shape=(*image_size, ch_in), name='input')
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    f3 = vgg16.get_layer('block3_pool').output
    f4 = vgg16.get_layer('block4_pool').output
    f5 = vgg16.get_layer('block5_pool').output
    f5_conv1 = Conv2D(filters=4086, kernel_size=7, padding='same', activation='relu')(f5)
    f5_drop1 = Dropout(0.5)(f5_conv1)
    f5_conv2 = Conv2D(filters=4086, kernel_size=1, padding='same', activation='relu')(f5_drop1)
    f5_drop2 = Dropout(0.5)(f5_conv2)
    f5_conv3 = Conv2D(filters=ch_out, kernel_size=1, padding='same', activation=None)(f5_drop2)
    f5_conv3_x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2,
                                  use_bias=False, padding='same', activation='relu')(f5)
    f4_conv1 = Conv2D(filters=ch_out, kernel_size=1, padding='same', activation=None)(f4)
    merge1 = add([f4_conv1, f5_conv3_x2])
    merge1_x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')(merge1)
    f3_conv1 = Conv2D(filters=ch_out, kernel_size=1, padding='same', activation=None)(f3)
    merge2 = add([f3_conv1, merge1_x2])
    outputs = Conv2DTranspose(filters=ch_out, kernel_size=16, strides=8, padding='same', activation=None)(merge2)
    fcn_model = Model(inputs, outputs)
    return fcn_model


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = fcn(image_size=(128, 128))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions(data)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8)
EPOCHS = 50
VAL_SUBSPLITS = 5
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          callbacks=[reduce_lr, early_stopping])
