# Denoising-CNN-AutoEncoder
A PyTorch CNN Autoencoder trained to remove static noise from audio files

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The inspiration for this project came from the countless whrrrs and hisses coming from my favourite songs that would never go away with conventional denoising. Part of why this problem is so hard and no real high fidelity solutions exist is that white noise covers the full range of frequencies, putting an almost film grain over the entirity of the audio's spectrogram. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

My solution? A convolutional autoencoding neural network. The algorithm begins by transforming the audio into a number representation by using the short-time Fourier transform, making it into a human-understandable spectrogram image.
From there we detect noise using filters which work by calculating the dot product between a 3 x 3 filter and the 3 x 3 sections of the image. The resulting output is used as the pixel values for the image thus compressing it as the filter traverses the spectrogram. The values in these filters are trained through backpropogation to detect the specific patterns unwanted noise has in the audio. The outputted values are then passed to an activation function called ReLu, this function applies max(0, x) to all values in the spectrogram and allows for more distinct states of 'activation' in the network. This particular model has three compressing stages in the encoder outputting a relatively large sized latent space which is then plugged into a decoder designed as the exact opposite of the encoder to transform the latent space back into a hopefully denoised version of our spectrogram.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Vocaroo links for sample output (non-copyrighted):  
Notes:  
MAY BE LOUD!  
This song was not used in training so this is a more true test of the model's performance  

Noisy input:  
https://voca.ro/19e5S1FPDlNj

Cleaned output:  
https://voca.ro/1b3zsW87y7nY

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Instructions:    
  
Making predictions with the model  
Notes:  
The model will run substantially faster with a CUDA enabled NVIDIA GPU  
The length of audio which the model can handle will be depandant on how much system memory/VRAM your computer has equipped
       
1. set up a Python 3.9 environment
2. run the following commands to set up needed libraries:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   pip install -r requirements.txt
3. in 'Audio_Denoiser.ipynb' set variable 'input_song' to the path of your input song then run all cells!  
  
  
Training the model:  
Notes: training will require a CUDA enabled NVIDIA GPU  
  
1. follow all above set up steps
2. use file 'Audio_Training_Data_Gen.ipynb' to create a data set by placing your training songs into a folder then passing the path to variable 'source_folder'
3. create a folder for the generated training data and set it to variable 'output_folder'
4. run all cells, and both a clean and Gaussian noised version of your input files will be generated in neat folders
5. in 'Audio_Denoiser.ipynb' set train = True and adjust other hyperparameters if needed
6. Run all cells! 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Next steps:  
1. Architecture:  
The overarching architecture of the model simply modifies images which happen to be audio, as such we can use many different architectures to achieve the same result. I've experimented with Variational CNN autoencoders but ran into memory   issues as nn.Linear() layers connected to every number in the latent space requires massive computation and RAM resources. Alternatively NVIDIA wrote a promising paper on their 'StyleGAN' architecture which people have been having success with and I want to further research and implement into my denoising application.

2. Dataloading:  
Currently the model was trained in a suboptimal way using transfer learning from previous training data batches running thousands of epochs each time. This leaves the model prone to learning the training data too well to make predictions on other songs or 'overfitting'. By using standard data loading tools the training data would be shuffled in thus preventing overfitting.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
