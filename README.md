## Abstract
In this paper, we propose an end-to-end model for generating encoding vectors for sentences describing an image. We use the encoded vectors from our network to generate images using Stacked Generative Adversarial Networks to generate high resolution quality images. We present different model architectures with word and character level representations and provide an extensive qualitative and quantitative evaluation on images generated. 

Full project report can be found here.

## Main architecture
In this paper, we propose an end-to-end hybrid CNN-RNN model to learn the relation between an image and its text description. The learned model is then used to train Stack GAN conditioned on an input caption, whose encodings are generated from our trained model.
![alt text](MainArchitecture.png?raw=true "Title")

