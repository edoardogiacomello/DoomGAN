### DCGAN-slerp-interpolation
This folder contains a modification of the code present at https://github.com/carpedm20/DCGAN-tensorflow
and https://github.com/dribnet/plat for generating spherical interpolated samples from the CelebA dataset as seen in https://arxiv.org/abs/1609.04468.

In particular, the net has been trained for 8 epochs then sampled with two random seeds (42 and 314) and the latent spaces of the saples are
spherically interpolated leading to the [following result](https://drive.google.com/open?id=0B9y_HgFPj7_ra19ONnQ1VmFwN3c).
[Another example](https://drive.google.com/open?id=0B9y_HgFPj7_rUlNydC1xRjZCMW8) with seed_one=68384 and seed_two=86554

Moreover, I also added a simple example of "face algebra":
Addends (seed68384, seed86554):
![First Addend: seed68384](https://github.com/edoardogiacomello/DoomPCGML/blob/master/GAN-Tests/DCGAN-slerp-interpolation/samples/seed:68384.png?raw=true)
![Second Addend: seed86554](https://github.com/edoardogiacomello/DoomPCGML/blob/master/GAN-Tests/DCGAN-slerp-interpolation/samples/seed:86554.png?raw=true)
Sum (seed68384 + seed86554):
![Sum](https://github.com/edoardogiacomello/DoomPCGML/blob/master/GAN-Tests/DCGAN-slerp-interpolation/samples/seed:68384%2Bseed:86554.png?raw=true)
Subtraction (seed68384 - seed86554):
![Sub](https://github.com/edoardogiacomello/DoomPCGML/blob/master/GAN-Tests/DCGAN-slerp-interpolation/samples/seed:68384-seed:86554.png?raw=true)

####Usage:
This command launches both the interpolation and the sample algebra:
```
python main.py --dataset celebA --input_height=108 --crop --seed_one=68384 --seed_two=86554
```
You can change freely the two seeds in order to obtain different results.

This experiment has been made merely for testing my understanding of the techniques presented in the papers cited above rather than improving the network.
The dataset is not included (see the original readme for detail on how to download it and train the net) but a pre-trained model is.


