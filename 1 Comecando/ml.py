import tensorflow as tf
import numpy as np
from tensorflow import keras

# Uma rede neural que usa apenas um neurônio
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Determina os modelos matemáticos usados para calcular o quão distante da resposta
# a tentativa está, e como otimizar para a próxima tentativa
model.compile(optimizer='sgd', loss='mean_squared_error')

# Dados para aprender
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Roda o modelo para treinamento
model.fit(xs, ys, epochs=500)