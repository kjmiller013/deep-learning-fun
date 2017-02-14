# deep-learning-fun
For fun, useless projects only :)

#Project 1 - Feline ==> English:
##Game Plan:
1.) Obtain lots of cat videos.  Store their titles and frames.
2.) Find an already-trained convnet that does pose estimation.  Run it on the cat frames and store the poses as vectors (or next to last layer as vector).  Use a human pose-net if that's all that's available - it'll probably pick up some kind of information about the cat poses.
3.) Find an already-trained semantic-vector-space model that can encode and decode between phrases and phrase-vectors.
4.) Design a simple LSTM that takes in the pose-vectors and spits out a phrase-vector at the end.  Use the most lightweight one you can find!  Make the hidden state as small as you want and have a single layer that converts the final state into a phrase-vector.  Use ReLUs whenever possible!
5.) Train the LSTM (and single layer at the end) to match the title phrase-vector given the sequence of pose-vectors.
6.) At test-time, decode the predicted phrase vector to get your "translation".

This is basically a poor man's video captioning, except that instead of video we only let it look at the cat poses, so it can only "caption" the kitty.
