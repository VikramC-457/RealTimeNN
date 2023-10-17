import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import sys
from networktables import NetworkTables
from tensorflow import keras
import time
import numpy as np

epoch = 300
n = 100 #len of total dist to target
duration = 10
learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
def connectionListener(connected, info):
    print(info, f"; Connected={connected}")

# Initialized NT with RoboRIO IP
IP = sys.argv[1]
NetworkTables.initialize(server=IP)
# Tells if NT connectes to the RoboRIO
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)
NetworkTables.setUpdateRate(50) # ms
def lossfunction(predictions,distX, crash, distY):
    # Ensure that all inputs are TensorFlow tensors

    # Calculate the loss using the provided formula
    XPower,YPower = predictions[0,0],predictions[0,1]
    speed = tf.sqrt(tf.square(XPower)+tf.square(YPower))
    
    term1 = tf.add(1.0, tf.add(tf.multiply(5.0, crash), tf.multiply(5.0, distY)))
    term2 = tf.add(distX, speed)
    loss = tf.subtract(tf.multiply(n, term1), term2)

    return loss
def returnData(predictions):
    Xpower,Ypower=float(predictions[0,0]),float(predictions[0,1])
    NetworkTables.getTable('Drive').getEntry('speedValueX').setDouble(Xpower)
    NetworkTables.getTable('Drive').getEntry('speedValueY').setDouble(Ypower)
    # Replace with your actual implementation
    # Send commands to the robot within this function
    return None

def getData():
    # Replace with your actual data source or loading method
    distFront=NetworkTables.getTable('Drive').getValue(key='distance',defaultValue=2)
    distBack=NetworkTables.getTable('Drive').getValue(key='distance',defaultValue=2)
    distLeft=NetworkTables.getTable('Drive').getValue(key='distance',defaultValue=2)
    distRight=NetworkTables.getTable('Drive').getValue(key='distance',defaultValue=2)
    
    data=[distFront,distBack,distLeft,distRight]
    data = np.asarray(data)
    #print(data)
    return tf.convert_to_tensor(data.reshape(1, -1), dtype=tf.float32)


def create_model():
    #model is mildly complex, maybe too complex for simple A->B goal type, and probably too big for ras pi, will tune
    input_tensor = Input(shape=(4,))
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

    
    distX = float(input("distx "))
    #distX = tf.constant(distX, dtype=tf.float32)
    crash = int(input("crash "))
    #crash = tf.constant(crash, dtype=tf.float32)
    #speed = tf.constant(speed, dtype=tf.float32)
    distY = float(input("distY "))
    loss=tf.Variable(0.0)
    #distY = tf.constant(distY, dtype=tf.float32)
    with tf.GradientTape() as tape:
        while time.time() - start_time < duration:
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
