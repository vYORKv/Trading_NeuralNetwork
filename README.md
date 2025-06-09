# Day Trading Neural Network

Basic neural network that trains on S&P 500 futures data for day trading. 

This neural net outputs a json model that can be plugged directly into Quantower trading platform strategies and execute trades through the strategy script.

**Plans for this project**:
- Will rewrite this model in C++ both to practice my C++ programming and optimize for performance.
- Intend to evolve this base model into an LSTM re-current neural network.
- Will add a Quantower strategy script to this repo that allows the model to execute trades in Quantower.

# Data Disclaimer
The market data in this project was purchased from a data broker in its initial form. I have since reformatted and modified the data in such a way that you can legally use the data for your own projects. Feel free to make use of this data with full permissions.