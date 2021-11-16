from absl import app
import tensorflow as tf
import numpy as np
import os
import time
from devfacPyLogger import log

data_dir = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

batch_size = 64
seq_length = 100
embedding_dim = 256
hidden_size = 1024
num_epochs = 10

text = open(data_dir, 'rb').read().decode('utf-8')
vocab = sorted(set(text))
vocab_size = len(vocab)
log.info(f'unique characters:{vocab_size}')
char2idx = { u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

# seq +1 : chunk 한칸을 밀어야 targeT_text를 넣을 수 있다.
sequence = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequence.map(split_input_target)

dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)


class RNN(tf.keras.Model):
  def __init__(self, batch_size):
    super(RNN, self).__init__()
    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
    self.hidden_layer_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')
    self.output_layer = tf.keras.layers.Dense(vocab_size)

  def call(self, x):
    embedded_input = self.embedding_layer(x)
    features = self.hidden_layer_1(embedded_input)
    logits = self.output_layer(features)

    return logits
  
def sparse_cross_entropy_loss(labels, logits):
  return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

optimizer = tf.optimizers.Adam()

@tf.function
def train_step(model, input, target):
  with tf.GradientTape() as tape:
    logits = model(input)
    loss = sparse_cross_entropy_loss(target, logits)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss

def generate_text(model, start_string):
  num_sampling = 40000

  # 시작 integer를 찾아 변환
  input_eval = [char2idx[s] for s in start_string]
  # 왜 차원을 늘려주지?
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []
  temperature = 1.0

  model.reset_states()
  for i in range(num_sampling):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


def main(_):
  RNN_model = RNN(batch_size=batch_size)
  
  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_preditions = RNN_model(input_example_batch)
    log.info(f'{example_batch_preditions.shape} #(batch_size, sequence_length, vocab_size)')
  
  RNN_model.summary()
  
  checkpoint_dir = './rnn_traning_checkpoint'
  # checkpoint_prefix = os.path.join(checkpoint_dir, f'ckpt_{epoch}')
  
  for epoch in range(num_epochs):
    start = time.time()
    checkpoint_prefix = os.path.join(checkpoint_dir, f'ckpt_{epoch}')
    
    # 왜 초기화 하지?
    hidden = RNN_model.reset_states()
    
    for (batch_n, (input, target)) in enumerate(dataset):
      loss = train_step(RNN_model, input, target)
      
      if batch_n % 100 == 0:
        log.info(f'Epoch {epoch} Batch {batch_n} Loss {loss}')
        
    if (epoch+1) % 5 == 0:
      RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))
    
    log.debug(f"Epoch {epoch+1} /Loss : {loss} /걸린시간: {time.time() - start}초")
    
  # RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))
  log.debug('트레이닝이 끝났습니다.')
  
  sampling_RNN_model = RNN(batch_size=1)
  sampling_RNN_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  sampling_RNN_model.build(tf.TensorShape([1, None]))
  sampling_RNN_model.summary()
  
  log.debug('샘플링을 시작합니다!')
  log.info(generate_text(sampling_RNN_model, start_string=u' '))
  

if __name__ == '__main__':
  # main 함수를 호출합니다.
  app.run(main)
  
  