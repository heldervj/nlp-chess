# Mastering Chess Through Neural Languages

# WIP

Author: Helder Vieira

Abstract:

Natural language tasks in Machine Learning are dominated by models that implement the transformer architecture, such as BERT, T5 and GPT-2. There is no doubt that this approach fits excellently well to perform tasks such as translation, text generation, sentiment analysis, among others traditional cases. The goal of this experiment is to measure, in a qualitative way, the capability of these models to generalize complex rules and environments. To achieve this, we used a well-known deterministic closed domain: the chess game. We trained with 1 epoch a GPT-2 model with 99M parameters in a dataset with about 2.2M chess games in the PGN representation, where the game is represented by a sequence of moves (tokens) and read from left to right. The study suggested that the model was capable of performing a reasonable generalization, playing well known opening positions or usual movements, but struggling with basic out-of-the-standard positions, thus being incapable of beating a human player.


# To reproduce:

All the experiment was made in a Google Colab environment.

Download the dataset used for training here: https://zenodo.org/record/5767528#.YbGYjvHMLDI

### First

Run the src/ETL_PGN.ipynb

Remember to change the references to the file.

### Second

Run the src/Generate_Train_Dataset.ipynb

Again, pay attention to the paths.

### Third

Run the src/Chess_NLP_Training_and_Playing.ipynb

There are 2 versions of the model: with and without the pre-movements sequence. Feel free to try it out. There's some classes at the bottom of the notebook that you can use to play againt the model.

# TODO

- Clean the code: there is a lot of code in portuguese;
- Remove the code from the notebooks and build a reasonable repo;
- Send trained models to HuggingFace;
- I have an old code with python-chess that I took as base to build the playing environment. I still didn't find the blog post that implemented it at first. As soon as I found it, I'll put the reference here.