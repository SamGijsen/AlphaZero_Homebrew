### AlphaZero Homebrew

Homebrew implementation of AlphaZero using NumPy/PyTorch for TicTacToe


The algorithm leverages the following pieces:
* `mcts.py`: Monte Carlo tree search, used to plan ahead by simulating game roll-outs.
* `neural_nets.py`: Includes the PyTorch models. (Note: The original publication uses only a single network.)
  * Policy net: Outputs a probability vector over possible moves given a game state. Used by MCTS to provide a prior distribution.
  * Value/Target net: Outputs a Q-value for a given game state. Used by MCTS to evaluate a state of a game that is not (yet) completed.
* `self_play.py`: Uses MCTS to run games and logs all activity for evaluation and training.
* `TTT_env.py`: TicTacToe environment.

The resulting `Self_Play` and `Training` classes allow for a recursive process: first, a batch of games is played, which afterwards serve as training data for the DNNs. If all goes well, the trained DNNs allow for better play on the next batch of games, and so on. This play-training loop is implemented in the notebook. Here is a preview:
```py

for v in range(iterations):
    
    # start with self-play
    print("Self-Play: Iter {} out of {}".format(v+1, iterations))

    engine = Self_Play(games=games, depth=depth, temperature=temperature, parameter_path = parameter_path)
    state_log, mcts_log, win_log = engine.play(version=v)
    
    
    # train DNN's using the played games
    print("Train: Policy & Value: Iter Net {} out of {}".format(v+1, iterations))
    
    if v == 0:
        train = Training()   
        
    pnet, losses = train.train_policy( 
    state_log, mcts_log, win_log, version=v, parameter_path = parameter_path, lr=lr_p, batchsize=batchsize_p, epochs=epochs_p
    )
    vnet, losses = train.train_value( 
    state_log, mcts_log, win_log, version=v, parameter_path = parameter_path, lr=lr_v, batchsize=batchsize_v, epochs=epochs_v
    )
    
```

![image](https://user-images.githubusercontent.com/44772298/178162295-e4ac5bf0-3a6e-42a0-90da-1cfcf5001c40.png)
