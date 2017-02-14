# deep-learning-fun
For fun, useless projects only :)

#Project 1 - Feline ==> English:
##Game Plan:
####1.) Obtain lots of cat videos.  Store their titles and frames.
####2.) Use something like https://github.com/shekkizh/FCN.tensorflow to get a semantic segmentation of the video and get a kitty mask from that.  Compute a bounding-box around the kitty and crop out just that part of the mask and resize it and flatten into a "silhouette-vector".  For now don't even try to do anything else to the silhouette.  If there's no kitty, just output all zeros.
####3.) Find an already-trained semantic-vector-space model that can encode and decode between phrases and phrase-vectors.  Something like char-rnn.
####4.) Design a simple LSTM that takes in the silhouette-vectors and spits out a phrase-vector at the end.  Use the most lightweight one you can find!  Make the hidden state as small as you want and have a single layer that converts the final state into a phrase-vector.  Use ReLUs whenever possible!
####5.) Train the LSTM (and single layer at the end) to match the title phrase-vector given the sequence of silhouette-vectors.
####6.) At test-time, decode the predicted phrase vector to get your "translation".
####
####This is basically a poor man's video captioning, except that instead of video we only let it look at the cat silhouettes, so it can only "caption" the kitty.  Between the silhouette and the phrase vector there's only one hidden layer (the hidden state of the LSTM), but technically one hidden layer is all you need for a neural net to be a universal approximator, so...
