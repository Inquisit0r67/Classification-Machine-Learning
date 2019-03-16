# Classification-Machine-Learning

### Five key steps in a machine learning project life cycle 

- Data collection 

  - This will be the data points that your ML model will make predictions on, for this project I will use an existing proprietary data set which consists of  28x28 *NumPy* arrays, with pixel values ranging between 0 and 255. The *labels* are an array of integers, ranging from 0 to 9.
  - This ML project will train a neural network model to classify images of clothing, like sneakers and shirts.

- Data normalisation

  - In this project this involves getting the pixel values down to between 0 and 1:

    - ```python
      train_images = train_images / 255.0
      test_images = test_images / 255.0
      ```

       We then evaluate the data the ensure it is ready to be used:

      ```python
      plt.figure(figsize=(10,10))
      for i in range(25):
          plt.subplot(5,5,i+1)
          plt.xticks([])
          plt.yticks([])
          plt.grid(False)
          plt.imshow(train_images[i], cmap=plt.cm.binary)
          plt.xlabel(class_names[train_labels[i]])
      plt.show()
      ```

- Data Modelling / Building the model

  - I will be using This guide uses [tf.keras](https://www.tensorflow.org/guide/keras), a high-level API to build and train models in *TensorFlow*. More specifically the `keras.sequential` model. The basic building block of a neural network is the *layer*. Layers extract representations from the data fed into them and use those representations. Deep learning consists of chaining multiple layers together 

  - The first layer in this network, ` keras.layers.FLatten() ` , transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.

  - After the pixels are flattened, the network consists of a sequence of two `keras,layers.Dense` layers. These are densely-connected, or fully-connected, neural layers. The first `Dense` layer has 128 nodes (or neurons). The second (and last) layer is a 10-node SoftMax layer—this returns an array of 10 probability scores that sum to 1.

    ```python
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    ```

  - We then compile the model which adds a few more setting crucial to the ML model.

    - *Loss function* —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.

    - *Optimizer* —This is how the model is updated based on the data it sees and its loss function.

    - *Metrics* —Used to monitor the training and testing steps. The following example uses *accuracy*, the fraction of the images that are correctly classified.

      ```python
      model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
      ```

- Model Training

  - After the input as been prepared we can begin to train our ML model, we separate the data set into inputs we use to train the model and inputs we use to validate the model. In larger datasets a sample is usually taken. A key component of this phase is to iterate rapidly, continuously testing new data points that can be derived from the data source. This process is called *feature engineering*.

    ```python
    model.fit(train_images, train_labels, epochs=5)
    ```

- Model deployment

  - This one wont be done as this is merely a test using a "hello world" data set



More detailed overview of this project can be found on my website: https://inquisit0r67.github.io