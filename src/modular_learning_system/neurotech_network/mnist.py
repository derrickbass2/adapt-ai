from tensorflow.keras import models, layers


class MNISTClassifier:
    def __init__(self):
        self.model = None

    def build_and_train(self, train_data, val_data):
        """
        Build and train a simple neural network model for MNIST classification.

        Parameters:
            train_data (tuple): Tuple containing training images and labels.
            val_data (tuple): Tuple containing validation images and labels.
        """
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Define a simple model
        self.model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Train the model
        self.model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))

    def load_model(self, model_path: str):
        """
        Load a pre-trained model from a specified path.

        Parameters:
            model_path (str): Path to the saved model.
        """
        self.model = models.load_model(model_path)

    def predict(self, data):
        """
        Predict the class for given data using the trained model.

        Parameters:
            data: Input data for prediction.

        Returns:
            Predicted classes.
        """
        return self.model.predict(data)
