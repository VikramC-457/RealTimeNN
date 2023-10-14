import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from networktables import NetworkTables
from tensorflow import keras
import time
import numpy as np

epoch = 300
n = 1000
duration = 10
learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

def lossfunction(predictions,distX, crash, distY):
    # Ensure that all inputs are TensorFlow tensors

    # Calculate the loss using the provided formula
    XPower,YPower = predictions[0,0],predictions[0,1]
    speed = tf.sqrt(tf.square(XPower)+tf.square(YPower))
    
    term1 = tf.add(1.0, tf.add(tf.multiply(5.0, crash), tf.multiply(5.0, distY)))
    term2 = tf.add(distX, speed)
    loss = tf.subtract(tf.multiply(n, term1), tf.multiply(2.0, term2))

    return loss
def returnData(input_data):
    # Replace with your actual implementation
    # Send commands to the robot within this function
    return None

def getData():
    # Replace with your actual data source or loading method
    distFront=NetworkTables.getTable('distFront')
    distBack=NetworkTables.getTable('distBack')
    distLeft=NetworkTables.getTable('distLeft')
    distRight=NetworkTables.getTable('distRight')
    data=[distFront,distBack,distLeft,distRight]
    data = np.asarray(data)
    #print(data)
    return tf.convert_to_tensor(data.reshape(1, -1), dtype=tf.float32)


def create_model():
    input_tensor = Input(shape=(5,))
    x = Dense(256, activation='relu', trainable=True)(input_tensor)
    x = Dense(512, activation='relu', trainable=True)(x)
    x = Dense(512, activation='relu', trainable=True)(x)
    x = Dense(256, activation='relu', trainable=True)(x)
    x = Dense(2, activation='tanh', trainable=True)(x)
    model = Model(inputs=input_tensor, outputs=x)
    model.summary()
    return model

model = create_model()

for i in range(epoch):
    print("\nStart of epoch %d" % (i,))
    start_time = time.time()

    while time.time() - start_time < duration:
        distX = float(input("distx "))
        #distX = tf.constant(distX, dtype=tf.float32)
        crash = int(input("crash "))
        #crash = tf.constant(crash, dtype=tf.float32)
        #speed = tf.constant(speed, dtype=tf.float32)
        distY = float(input("distY "))
        #distY = tf.constant(distY, dtype=tf.float32)

        with tf.GradientTape() as tape:
            input_data = getData()
            predictions = model(input_data, training=True)
            loss = lossfunction(predictions,distX, crash, distY)
            print("TrainingLoss: " + str(loss))

            # Execute commands and send them to the robot
            returnData(predictions)

        # Compute gradients and update model weights
        gradients = tape.gradient(loss, model.trainable_weights)
        #print("Gradients: ", gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
