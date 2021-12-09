# Mastering Chess Through Neural Languages

For more details, read de pdf file.

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


# Games played with the model, I was always with the white pieces

I'm not good at chess.

Model Without Pre-Movements

Game 1

d4 Nf6 Nf3 g6 Bf4 Bg7 e3 O-O Bd3 d6 O-O Nbd7 c3 Re8 Qe2 e5 dxe5 dxe5 Bg3 e4 Bxe4 Nxe4 Bf4 Qe7 Nbd2 Ndf6 Nxe4 Nxe4 h3 Bf5 g4 Bd7 Rad1 Bc6 Nd2 Nxd2 Rxd2 Rad8 Rfd1 Rxd2 Qxd2 Rd8 Qxd8 Qxd8 Rxd8 Bf8 Bxc7 Kg7 Bd6 f5 Bxf8 Kf6 gxf5 gxf5 Bc5 a6 Rf8 Ke6 Rh8 Kd5 Bd4 Ke4 Rxh7 Kd3 Rh5 Kc2 Rxf5 Kxb2 h4 Kxa2 h5 Kb3 h6 a5 h7 a4 h8=Q a3 Ra5 b5 Rxa3 Kc4 Qg8+ Kd3 Qc8 Ke4 Qxc6+ Kd3 Qxb5+ Ke4 c4 Kf3 Qf5+ Ke2 Ra2+ Ke1 Qb1

Game 2

e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Bd3 Nc6 Nxc6 bxc6 O-O e5 Bg5 Be7 Nc3 O-O Bc4 Be6 Bxe6 fxe6 Re1 Qc7 h3 Rad8 b4 d5 exd5 cxd5 Rb1 d4 Ne4 Nxe4 Rxe4 Bxg5 h4 Bh6 g4 Bf4 f3 Qf7 h5 h6 Qe1 Bg5 Qf2 Bf4 Rb3 Rc8 Qe1 Rxc2 b5 d3 Qd1 d2 b6 axb6 Rxb6 Rc1 Qxc1 d1=Q+ Qxd1 Be3+ Rxe3 Qxf3 Rxf3 Rxf3 a4 Ra3 Rxe6 Rxa4 Qd8+ Kh7 Qe8 Rf4 Qg6+ Kh8 Re8+ Rf8 Rxf8#

Game 3

g4 d5 Bg2 e5 f3 Bc5 h4 h5 d3 hxg4 fxg4 Bxg4 Bf3 Bxf3 Nxf3 e4 dxe4 dxe4 Qxd8+ Kxd8 Ng5 Nh6 Nxe4 Bb6 c4 Nf5 Ng5 Nc6 Nxf7+ Ke7 Nxh8 Rxh8 Bg5+ Ke6 Nc3 Ne5 Kd2 Nxc4+ Kc2 Nd4+ Kd1 Ne3+ Kc1 Nc4 Be3 Nxe3 Nd1 Nxd1 Rxd1 Nf5 b3 Ne3 Rd3 Ng4 Kb2 Ne5 Rc3 c6 Rg3 Kf5 Rxg7 Rxh4 Rxb7 Rh2 Rf1+ Ke6 Re1 Kd6 Ka3 a5 Rb8 Kc7 Re8 a4 Rxe5 axb3 axb3 c5 b4 cxb4+ Kxb4 Kc6 Rc1+ Kb7 Kb5 Ka7 Re7+ Ka8 Rc8#

Game 4

e4 c5 Ne2 Nc6 d4 cxd4 Nxd4 Nf6 Be2 e5 Nf5 d5 O-O Bxf5 exf5 d4 Bg5 Be7 c3 O-O cxd4 exd4 Bg4 Nxg4 Qxg4 Bxg5 h4 Bf6 Nd2 Re8 Rfe1 Ne5 Qe4 Qb6 b3 Rad8 Rac1 d3 Kh2 Qd4 Qxd4 Rxd4 Re4 Rxe4 Nxe4 Bxh4 g3 Be7 Rc7 Kf8 Rxb7 a5 a3 h5 b4 axb4 axb4 Bxb4 Rxb4 d2 Nxd2 Nf3+ Nxf3 Re2 Rb8+ Ke7 Rb7+ Kf6 Kh3 Rxf2 Rb6+ Kxf5 Rb5+ Kg6 Ne5+ Kf6 Ra5 g6 Kh4 Rf5 Nc6 Kg7 Rxf5 gxf5 Kxh5 Kf6 Nd4 Ke5 Nf3+ Kf6 Nh4 Kg7 Nxf5+ Kf6 g4 Ke5 Nh6 f6 Ng8 Ke6 Nxf6 Kxf6 g5+ Kg7 g6 Kg8 Kh6 Kh8 g7+ Kg8 Kg6

Game 5

d4 Nf6 e3 g6 Nc3 Bg7 Bd2 O-O Qe2 d6 O-O-O c6 g3 b5 f4 b4 Nb1 a5 Bg2 Ba6 Nh3 Qb6 Ng5 d5 e4 dxe4 Nxe4 Nxe4 Qxe4 Nd7 Qxe7 Nf6 d5 cxd5 Be3 Qb7 Bc5 Rfe8 Qxb7 Bxb7 Rhe1 Ne4 Nd2 Nxc5 Rxe8+ Rxe8 Nf3 Re2 Rd2 Rxd2 Kxd2 Ne6 Ne5 Bxe5 fxe5 Kf8 c3 bxc3+ Kxc3 Ke7 b4 axb4+ Kxb4 Ba6 a4 Kd7 a5 Kc6 h4 h5 Bh3 d4 g4 hxg4 Bxg4 d3 h5 gxh5 Bxh5 d2 Kc3 Kd5 Kxd2 Kxe5 Bxf7 Nd4 Kc3 Nb5+ Kb4 Nc7 Bc4 Kd6 Bxa6 Nxa6+ Kb5 Nc7+ Kb6 Nd5+ Kb7 Ne7 a6 Nc6 Kb6 Nb8 a7 Nd7+ Kb7 Nc5+ Kb8 Nd7+ Kc8 Kc6 a8=Q+ Kd6

Game 6

d4 Nf6 c4 e6 Bd2 d5 cxd5 exd5 Qa4+ c6 e3 Bd6 Ba5 Qe7 Nc3 O-O O-O-O b5 Qc2 b4 Nce2 Ba6 b3 Bxe2 Bxe2 c5 dxc5 Bxc5 g3 Nc6 Nf3 Rac8 Nh4 g6 f3 Ne5 Rhe1 Rfe8 h3 Qb7 f4 Ned7 g4 Ne4 Qb1 Nc3 Qb2 Nxd1 Rxd1 Nf6 Qc2 Ne4 f5 Nc3 Rd2 Nxe2+ Kb2 Nd4 Qd3 Re4 fxg6 hxg6 Ng2 Ne6 g5 d4 exd4 Nxd4 Ne1 Qc6 h4 Nf5 Re2 Nxh4 Rxe4 Qxe4 Qxe4 Ng2 Nxg2 Re8 Qxe8+ Kg7 Qc8 Bb6 Bxb6 axb6 Qb8 b5 Qxb5 f5 Qxb4 f4 Qxf4 Kh7 Qf6 Kg8 Qxg6+ Kf8 a4 Ke7 a5 Kd7 a6 Kc7 Qh6 Kb8 Qh7 Ka8 g6 Kb8 g7 Ka7 g8=Q+ Kb6 Qb8+ Kc6 Qh6+ Kd7 Qb7+ Kd8 Qh8#



Model With Pre-Movements

Game 1

d4 Nf6 Bf4 g6 Nf3 Bg7 e3 O-O Bd3 d6 c3 Nbd7 O-O Re8 Na3 e5 dxe5 dxe5 Bg3 Nh5 Nd2 Nxg3 hxg3 Nc5 Bc4 Be6 Bxe6 Rxe6 b4 Nd7 Ne4 Qe7 f4 exf4 exf4 f5 Ng5 Re3 Re1 Rxe1+ Qxe1 Re8 Qxe7 Rxe7 Kf2 Nf6 Nb5 c6 Nxa7 Ne4+ Nxe4 Rxe4 Re1 Rxe1 Kxe1 Bxc3+ Kd1 Bxb4 Kc1 Ba3+ Kc2 Kf7 Kb3 Bd6 Kc4 Ke6 Kb3 Kd5 Ka4 b6 Nc8 Bc5 Kb3 b5 Kc3 b4+ Kd3 b3 axb3 Bg1 Ne7+ Kd6 Ng8 c5 Ke2 c4 bxc4 Kc5 Kf3 Kxc4 g4 fxg4+ Kxg4 Kd3 Nf6 h6 f5 gxf5+ Kxf5 Bd4 Kg6 Bxf6 Kxf6 Ke4 Kg6 Kf4 Kxh6 Kg4 Kg6 Kf4 Kh5 Kf5 g3 Kf6 g4 Kf7 g5 Kg7 g6 Kg8 Kh6 Kh8 g7+ Kg8 Kh5 Kh7 Kg5 Kg8 Kh5 Kh7 Kg5 Kg8

Game 2

e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7 O-O Nc6 d4 cxd4 Nxd4 Nf6 Re1 g6 Bg5 Bg7 Nxc6 bxc6 b3 O-O c4 h6 Bh4 g5 Bg3 Nh5 Re3 Nxg3 Rxg3 f5 Nd2 f4 Rh3 e5 Qh5 Qf7 Qxf7+ Rxf7 g3 fxg3 hxg3 Raf8 Rf1 Rf3 Nxf3 Rxf3 Kg2 Rd3 Rfh1 Kf7 b4 Ke6 c5 dxc5 bxc5 Bf8 Rc1 Kd7 f4 gxf4 gxf4 exf4 Rxd3+ Ke6 Ra3 Bxc5 Rh1 Ke5 Rxh6 Kxe4 Re6+ Kd5 Re8 a5 Kf3 a4 Kxf4 Kc4 Rxa4+ Kd5 Rd8+ Ke6 Re4+ Kf7 a4 Bb6 Rb8 Ba5 Re5 Bc3 a5 Bd4 a6 Ba7 Ra8 Bb6 a7 Bxa7 Rxa7+ Kg6 Ra6 Kf7 Rxc6 Kg7 Re7+ Kf8 Ra7 Kg8 Rc8#

Game 3

e4 c5 d3 Nc6 Nf3 g6 Be2 Bg7 O-O d6 Be3 Nf6 Nbd2 O-O c3 e5 Rc1 h6 d4 cxd4 cxd4 exd4 Nxd4 Nxd4 Bxd4 Be6 b3 Qa5 Rc2 Rac8 Qc1 Rxc2 Qxc2 Rc8 Qb2 b5 h3 a6 e5 dxe5 Bxe5 Nd7 Bxg7 Qxd2 Qxd2 Kxg7 Bg4 Bxg4 hxg4 Nf6 Qa5 Nxg4 Qxa6 Rc2 Qxb5 Rxa2 Rc1 Nf6 Rc8 Ra1+ Kh2 Ra2 Qb6 Rb2 Qd8 Rxb3 Qh8#

Game 4

g4 d5 Bg2 c6 e3 e5 Ne2 Bd6 h4 h5 g5 Bg4 Bh3 Bxh3 Rxh3 Nd7 d4 e4 c3 Ne7 Nd2 Qc7 c4 dxc4 Nxc4 Bb4+ Kf1 b5 a3 bxc4 axb4 Nd5 Bd2 O-O Rg3 f5 gxf6 Rxf6 Rg5 Raf8 Rxh5 Rf5 Rxf5 Rxf5 f4 exf3 Ng1 f2 Nh3 Rf3 Nxf2 Nxe3+ Bxe3 Rxe3 Qd2 Rd3 Qc2 Rxd4 Rd1 Rxd1+ Qxd1 Qf4 Qxd7 Qc1+ Kg2 Qxb2 Qxa7 c3 Qa8+ Kh7 Qxc6 c2 Nd3 Qd4 Ne1 Qxb4 Qxc2+ g6 h5 Qe4+ Qxe4 Kg7 Qxg6+ Kf8 Qh7 Ke8 Qa7 Kd8 h6 Kc8 h7 Kd8 h8=Q#

Game 5

e4 c5 Ne2 Nc6 d3 g6 g3 Bg7 Bg2 d6 O-O e6 Nd2 Nge7 b3 O-O c4 f5 Rb1 a5 Re1 e5 Bb2 f4 gxf4 exf4 Bxg7 Kxg7 Nxf4 Rxf4 f3 Ne5 Nf1 N7c6 Ng3 Qh4 Rb2 Nd4 Rf1 Bh3 Bxh3 Qxh3 Kh1 Raf8 Rd2 Qh4 Rg1 g5 Qf1 g4 Qd1 Qg5 Rgg2 h5 Rg1 h4 Ne2 Nxe2 Rxe2 h3 Rg3 Rxf3 Re1 Rf2 Re2 Rxe2 Qxe2 Rf3 Rg1 Rxd3 Qe1 Nf3 Qe2 Rd2 Rf1 Rxe2 Rd1 Nd4 Rf1 Nf3 Rd1 Nxh2 Rxd6 Nf3 Rd7+ Kg6 Rxb7 g3 Ra7 Rh2#
Game 6

d4 Nf6 c4 e6 Nc3 Bb4 Bd2 O-O f3 d5 Rc1 b6 cxd5 exd5 a3 Be7 e4 dxe4 fxe4 Bb7 e5 Nd5 Nxd5 Bxd5 Nf3 c5 dxc5 bxc5 Qc2 Nd7 e6 fxe6 Bc4 Nb6 Bd3 c4 Nd4 cxd3 Qxd3 e5 Nf5 e4 Qc3 Bf6 Qc7 Qxc7 Rxc7 Rf7 Rxf7 Kxf7 Rf1 g6 Ng3 Nc4 b3 Nxd2 Kxd2 Rc8 b4 Rc2+ Kxc2 e3 Re1 Bxg2 Rxe3 Bd5 Rd3 Ke6 a4 Be5 b5 h5 a5 h4 Ne2 g5 b6 axb6 axb6 g4 b7 Bxb7 Rb3 Bd5 Rb5 Bf3 Nc1 Bxh2 Rb6+ Kf5 Rb3 g3 Rxf3+ Kg4 Re3 h3 Re1 g2 Re4+ Kg3 Ne2+ Kf2 Kd1 Be5 Rxe5 h2 Rh5 h1=Q+ Rxh1 g1=Q+ Nxg1 Kg3


