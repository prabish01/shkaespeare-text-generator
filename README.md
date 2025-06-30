# Shakespeare Text Generator

A neural network-based text generator that creates Shakespeare-style text using LSTM (Long Short-Term Memory) networks. The model learns patterns from Shakespeare's works and generates new text with adjustable creativity levels through temperature sampling.

## Features

- **LSTM Neural Network**: Uses TensorFlow/Keras to build a character-level language model
- **Temperature Sampling**: Generate text with different levels of creativity and randomness
- **Pre-trained Model**: Includes functionality to save and load trained models
- **Flexible Text Generation**: Customizable output length and creativity parameters

## Requirements

```
tensorflow>=2.0
numpy
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/shakespeare-text-generator.git
cd shakespeare-text-generator
```

2. Install required dependencies:
```bash
pip install tensorflow numpy
```

## Usage

### Training a New Model

Uncomment the training section in the code to train a new model:

```python
# Uncomment the training blocks to train from scratch
# The model will be saved as 'text_generator.keras'
```

### Generating Text

Run the script to generate Shakespeare-style text with different temperature settings:

```bash
python shakespeare_generator.py
```

The script will output text samples at temperatures from 0.2 to 1.0, demonstrating different levels of creativity.

## How It Works

### Model Architecture
- **Input Layer**: Character-level one-hot encoded sequences
- **LSTM Layer**: 128 units for learning sequential patterns
- **Dense Layer**: Fully connected layer with softmax activation
- **Output**: Probability distribution over possible next characters

### Temperature Sampling
Temperature controls the randomness of text generation:
- **Low Temperature (0.2-0.4)**: More conservative, predictable text
- **Medium Temperature (0.5-0.7)**: Balanced creativity and coherence
- **High Temperature (0.8-1.0)**: More creative but potentially less coherent

### Training Process
1. Downloads Shakespeare text corpus from TensorFlow datasets
2. Preprocesses text (converts to lowercase, creates character mappings)
3. Creates training sequences of 40 characters predicting the next character
4. Trains LSTM model using categorical crossentropy loss
5. Saves trained model for future use

## Configuration

Key parameters you can adjust:

- `seq_length = 40`: Length of input sequences
- `step = 3`: Step size for creating training sequences
- `batch_size = 256`: Training batch size
- `epochs = 10`: Number of training epochs
- `learning_rate = 0.01`: RMSprop optimizer learning rate

## Output Examples

The generator produces text at different temperature settings:

**Temperature 0.2** (Conservative):
More repetitive, follows learned patterns closely

**Temperature 0.5** (Balanced):
Good mix of coherence and creativity

**Temperature 1.0** (Creative):
More experimental, potentially less coherent but more surprising

## Model Performance

- **Training Data**: ~500,000 characters from Shakespeare's works
- **Sequence Length**: 40 characters
- **Vocabulary Size**: Varies based on unique characters in corpus
- **Training Time**: Approximately 10-30 minutes depending on hardware

## Customization

### Using Your Own Text
Replace the Shakespeare corpus with your own text:

```python
# Replace this line:
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# With:
with open('your_text_file.txt', 'r') as f:
    text = f.read().lower()
```

### Adjusting Model Architecture
Modify the model for different performance:

```python
model.add(LSTM(256))  # Increase units for more complexity
model.add(LSTM(128, return_sequences=True))  # Add multiple LSTM layers
model.add(Dense(len(characters)))
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Shakespeare text corpus provided by TensorFlow
- Built with TensorFlow and Keras
-Inspired by NeuralNine 

## Future Improvements

- [ ] Add support for word-level generation
- [ ] Implement different RNN architectures (GRU, Transformer)
- [ ] Add fine-tuning capabilities
- [ ] Create web interface for text generation
- [ ] Add support for multiple authors/styles
