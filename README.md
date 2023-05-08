# Restaurant price category prediction
Authors: Marina Aziz, Anna Piroženoka 

TODO:
- implement baseline - perceptron? multiple perceptrons, each predicts one class category, pick highest probability
- feature preprocessing - separate class or under Corpus
- how to represent the features?
  - bag-of-words for menu items
  - one-hot encoding for location and food type
  - restaurant names - not sure yet? keywords like "pizza", "cafe" etc can be useful
- tokenization 
- how to store very sparse matrices/vectors - only store the indices

Done:
- evaluation methods
  - F-score
  - correlation score
