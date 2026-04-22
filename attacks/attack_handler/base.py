from common_imports import abc, np, pd, load_model, plt, itertools, tf

class Attack(abc.ABC):


    @abc.abstractmethod
    def apply(self, **kwargs):
        """
        Core attack logic. The meaning of parameters is up to subclass: 
        could be frames, dataframes, model, training data, etc.
        """
        pass


class EvasionAttack(Attack):
    @abc.abstractmethod
    def apply(self, frames: list[dict], labels: np.ndarray | None = None, **kwargs) -> list[dict]:
        """
        Perturb actual CAN frames or features to cause model misclassification.
        """

class GeneticAttack(Attack, abc.ABC):

    def __init__(self, model_path, file_path, population_size=100, max_generations=75, mutation_rate=0.1):

        self.model = load_model(model_path, compile=False)

        self.model.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.data = np.load(file_path)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.x_test = self.data['x_test']
        self.y_test = self.data['y_test']  
        

    def calculate_confidence(self, frame):

        frame_batch = np.expand_dims(frame, 0)
        prediction = self.model.predict(frame_batch, verbose=0)

        return prediction[0][1]

    def calculate_confidence_batch(self, population):
        """Predict confidence scores for an entire population in one call."""
        batch = np.array(population)
        predictions = self.model.predict(batch, verbose=0)
        return predictions[:, 1]


    def plot_confusion_matrix(self, cm, classes, suffix, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, filename=None):
        """
        Create and save a confusion matrix visualization with color coding and annotations.
        
        Args:
            cm: 2x2 confusion matrix array
            classes: List of class names ['Normal', 'Attack']
            suffix: String identifier for filename generation
            normalize: Whether to show percentages instead of raw counts
            title: Plot title
            cmap: Matplotlib colormap for visualization
            filename: Optional custom filename (auto-generated if None)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'  # Format: decimals for percentages, integers for counts
        thresh = cm.max() / 2.  # Threshold for text color (white on dark, black on light)
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if filename is None:
            filename = f"confusion_matrix_{suffix}.png"
        plt.savefig(filename)
        plt.close()

    @abc.abstractmethod
    def mutate(self, frame):
        raise NotImplementedError

    @abc.abstractmethod
    def crossover(self, parent1, parent2):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_adversarial_attack(self):
        raise NotImplementedError

