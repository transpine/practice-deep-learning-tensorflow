import tensorflow as tf

(x_train, y_train), (x_test, y_test ) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train / 255., x_test / 255.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000)
test_data_iter = iter(test_data)

class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
    self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
    self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
    self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
    self.conv_layer_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
    self.conv_layer_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
    self.conv_layer_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

    self.flatten_layer = tf.keras.layers.Flatten()
    self.fc_layer_1 = tf.keras.layers.Dense(384, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.2)

    self.output_layer = tf.keras.layers.Dense(10, activation=None)

  def call(self, x, is_training):
    h_conv1 = self.conv_layer_1(x)
    h_pool1 = self.pool_layer_1(h_conv1)
    h_conv2 = self.conv_layer_2(h_pool1)
    h_pool2 = self.pool_layer_2(h_conv2)
    h_conv3 = self.conv_layer_3(h_pool2)
    h_conv4 = self.conv_layer_4(h_conv3)
    h_conv5 = self.conv_layer_5(h_conv4)
    h_conv5_flat = self.flatten_layer(h_conv5)
    h_fc1 = self.fc_layer_1(h_conv5_flat)
    h_fc1_dropout = self.dropout(h_fc1, training=is_training)
    logits = self.output_layer(h_fc1_dropout)
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

@tf.function
def cross_entropy(logits, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


optimizer = tf.keras.optimizers.RMSprop(1e-3)

@tf.function
def train_step(model, x, y, is_training):
  with tf.GradientTape() as tape:
    y_pred, logits = model(x, is_training)
    loss = cross_entropy(logits, y)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def compute_accuracy(y_pred, y):
  correct_prediction = tf.equal( tf.argmax(y_pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

CNN_model = CNN()

for i in range(15000):
  batch_x, batch_y = next(train_data_iter)

  if i % 100 == 0:
    train_accuracy = compute_accuracy(CNN_model(batch_x, False)[0], batch_y)
    loss_print = cross_entropy(CNN_model(batch_x, False)[1], batch_y)

    print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
  
  train_step(CNN_model, batch_x, batch_y, True)

test_accuracy = 0.0
for i in range(10):
  test_batch_x, test_batch_y = next(test_data_iter)
  test_accuracy = test_accuracy + compute_accuracy(CNN_model(test_batch_x, False)[0], test_batch_y).numpy()
test_accuracy = test_accuracy / 10
print('정확도:%f' % test_accuracy)