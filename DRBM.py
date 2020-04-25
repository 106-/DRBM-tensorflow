import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import hidden_marginalize

class DRBM:
    def __init__(self, input_num, hidden_num, output_num, activation="continuous", dtype=tf.dtypes.float32, initial_sparse=10.):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.dtype = dtype

        self.w1 = tf.Variable( tf.keras.initializers.GlorotUniform()((input_num, hidden_num), dtype=self.dtype), name="w1" )
        self.w2 = tf.Variable( tf.keras.initializers.GlorotUniform()((hidden_num, output_num), dtype=self.dtype), name="w2" )
        self.b1 = tf.Variable( tf.zeros((hidden_num), dtype=self.dtype), name="b1" )
        self.b2 = tf.Variable( tf.zeros((output_num), dtype=self.dtype), name="b2" )
        self.params = [self.b1, self.b2, self.w1, self.w2]

        self.enable_sparse = "sparse" in activation
        if self.enable_sparse:
            self.sparse = tf.Variable( tf.cast(tf.fill([hidden_num], initial_sparse), dtype=self.dtype), name="sparse" )
            self.params.append(self.sparse)
        self._marginalize = getattr(hidden_marginalize, activation).activation

    # input: (N, i)
    # return (N, j, k)
    @tf.function
    def _signal_all(self, input):
        return tf.expand_dims(self.b1, 1) + self.w2 + tf.expand_dims(tf.matmul(input, self.w1), 2)

    @tf.function
    def probability(self, input, signal_value):
        sig = self._signal_all(input)
        signal_value(tf.reduce_max(sig))
        if self.enable_sparse:
            act = self._marginalize(sig, self.sparse)
        else:
            act = self._marginalize(sig)
        energies = self.b2 + tf.reduce_sum(act, 1)
        max_energies = tf.reduce_max(energies, axis=1, keepdims=True)
        return tf.nn.softmax(energies-max_energies)
    
    @tf.function
    def _negative_log_likelihood(self, probs, labels):
        single_prob = tf.reduce_sum(probs * labels, 1)
        return -tf.reduce_mean(tf.math.log(single_prob))

    def fit(self, train_epoch, optimizer, train_ds, test_ds):
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        signal_value = tf.keras.metrics.Mean(name='signal')

        for epoch in range(train_epoch):
            for images, labels in train_ds:
                self._train(images, labels, optimizer, train_loss, train_accuracy, signal_value)

            for test_images, test_labels in test_ds:
                self._test(test_images, test_labels, test_loss, test_accuracy, signal_value)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, signal_max: {}'
            print (template.format(epoch+1,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    test_loss.result(),
                                    test_accuracy.result()*100,
                                    signal_value.result()))
  
            # 次のエポック用にメトリクスをリセット
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            signal_value.reset_states()

    @tf.function
    def _train(self, input, labels, opt, train_loss, train_accuracy, signal_value):
        with tf.GradientTape() as tape:
            tape.watch(self.params)
            predict_probs = self.probability(input, signal_value)
            loss = self._negative_log_likelihood(predict_probs, labels)
        grads = tape.gradient(loss, self.params)
        opt.apply_gradients(zip(grads, self.params))
        train_loss(loss)
        train_accuracy(labels, predict_probs)

    @tf.function
    def _test(self, input, labels, test_loss, test_accuracy, signal_value):
        predict_probs = self.probability(input, signal_value)
        loss = self._negative_log_likelihood(predict_probs, labels)
        test_loss(loss)
        test_accuracy(labels, predict_probs)

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = (x_train / 255.0), (x_test / 255.0)

    dtype = tf.dtypes.float64

    x_train = x_train.reshape(-1, 784).astype(dtype.as_numpy_dtype())
    x_test = x_test.reshape(-1, 784).astype(dtype.as_numpy_dtype())
    y_train = to_categorical(y_train).astype(dtype.as_numpy_dtype())
    y_test = to_categorical(y_test).astype(dtype.as_numpy_dtype())

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002)
    #drbm = DRBM(784, 200, 10, activation="original", dtype=tf.dtypes.float64)
    drbm = DRBM(784, 200, 10, activation="continuous_sparse", dtype=tf.dtypes.float64)
    #drbm = DRBM(784, 200, 10, activation="continuous", dtype=tf.dtypes.float64)
    drbm.fit(500, optimizer, train_ds, test_ds)

if __name__=='__main__':
    main()
