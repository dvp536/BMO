# BMO
The AI

***IMPORTANT***  
The model trained (model.pkl) will only work with CUDA and not with the CPU.  
You will need to train it again if CUDA is not available.  

Libraries  
Numpy: 1.26.3 or lower  
SentencePiece: 0.2.0  
Torch: 2.5.1+cuxxx (Replace cuxxx for the cuda you have, for example cu118 is what we use)  
Torchaudio: 2.5.1+cuxxx (Replace cuxxx for the cuda you have, for example cu118 is what we use)  
Torchvision: 0.20.1+cuxxx (Replace cuxxx for the cuda you have, for example cu118 is what we use)
Transformers: 4.46.2  
  
**model.py**: The script that defines the model architecture and training process, including dataset loading, loss calculation, and optimization.  

**dataset.txt**: The file containing the raw data used to train the model, typically formatted as a series of dialogue examples or text.  

**BMO.py**: The script that loads the trained model and uses it to generate responses based on user input or prompts.  

**model.pkl**: The serialized file containing the trained model’s weights, generated after running the training process in model.py.  

**bpe.model**: The file storing the Byte Pair Encoding (BPE) model used during training to tokenize and encode the dataset for the model.  

**bpe.vocab**: The vocabulary file that maps tokens to their corresponding values, influencing token selection probabilities during text generation.  
