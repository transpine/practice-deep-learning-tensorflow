import tensorflow as tf

grad_clip = 5

class LinearRegression(tf.keras.Model):
  def __init__(self):
    super(LinearRegression, self).__init__()
    self.linear_layer = tf.keras.layers.Dense(1, activation=None)
    
  def call(self, x):
    y_pred = self.linear_layer(x)
    
    return y_pred
  

@tf.function
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred -y))

# optimizer = tf.optimizers.SGD(0.01, clip_by_norm=5)
optimizer = tf.optimizers.SGD(0.01)

@tf.function
def train_step(model, x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = mse_loss(y_pred, y)
    
  gradients = tape.gradient(loss, model.trainable_variables)
  
  clipped_grads = []
  for grad in gradients:
    clipped_grads.append(tf.clip_by_norm(grad, grad_clip))
  optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
  
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [2.0, 4.0, 6.0, 8.0]

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().batch(1)
train_data_iter = iter(train_data)

LinearRegressionModel = LinearRegression()

for i in range(1000):
  batch_xs, batch_ys = next(train_data_iter)
  
  batch_xs = tf.expand_dims(batch_xs, 0)
  train_step(LinearRegressionModel, batch_xs, batch_ys)
  
x_test = [3.5, 5.0, 5.5, 6.0]
test_data = tf.data.Dataset.from_tensor_slices((x_test))
test_data = test_data.batch(1)


for batch_x_test in test_data:
  batch_x_test = tf.expand_dims(batch_x_test, 0)
  print(tf.squeeze(LinearRegressionModel(batch_x_test), 0).numpy())

    