# Motion Magnification database


This repo contains python programs that generate video with motion magnification. 
It also creates the reference video with no motion magnification. 
The goal is to offer a database for quantitative evaluation of motion magnification methods.

### How it works

The videos are generated with a foreground image moving over a background.
We apply controlled geometric and translation transformations to the foreground images and put into the background.

##### Foreground
The foreground images were obtained from the alpha-mate dataset: http://www.alphamatting.com/datasets.php (a subset of the dataset is included in this repo).
From this dataset, we obtain 27 full images image and the alpha-mate mask that represent the pixels and how the semi transparent pixels blend with the background.

The transformations are applyed to the full image and also to the alpha-mate mask.
This addresses the quantization problem of sub-pixel transformations. The subpixel movement in the borders is viewed as a semi transparent intensity.

The full images are stored in the folder /input_training_highres and the alpha-mate images are stored at /gt_training_highres. The images present pixel width between 2080 and 3694; and pixels height between 1978 and 2600.
##### Background
The background images are from public domain and were taken from the pxhere.com webpage.
The images are high resolution landscape images with different proportions, presenting pixel width between 2199 and 9000; and pixels height between 1468 and 5304, and are stored in the folder /backgrounds

##### Foreground-background merge
The color of a pixel x is given by:
<img src="https://render.githubusercontent.com/render/math?math=c(x)=fg(x)*\alpha(x)+bg(x)*(1-\alpha(x))">,
where fg(x) if the foreground color at pixel x;
bg(x) is the background color at pixel x;
and <img src="https://render.githubusercontent.com/render/math?math=alpha(x)"> is the alha-mate semi transparency intensity (between 0 and 1) at pixel x.
Therefore, the foreground and background are blended toguether accordingly to the alpha-mate mask.

As the foreground and background image sizes may not match, the select as the output image as the foreground and resize and crop the background in order to fit.
The operation is performed at full image scale, but we downsize the image size to 1/4, in order to save space for saving the video.

##### Types of Movements
We prepare 2 types of movement. The first is a periodic translation, that can be in wither x,y and both axis. The file moveOonly.py generates a database of videos with only this type of movement.

The second is a periodic distortion. It alongs the object in one direction while retracts in the other periodically, like a bouncing ball. The file moveNdistort.py generates the database consisting of both types of movements.

##### Dataset 1: moveOnly
Created by the file moveOnly.py, it is composed by 27 videos (one for each foreground image) with motion amplification and other 27 corresponding to the videos with very slimm movement. Contains only periodic translation.

Each video has different setting, consisting in a combination of 3 factors:
* 3 different motion amplitudes;
* From the videos 2/3 has backgrounds from the background image data and 1/3 use a bland random color background; 
* Movement in X, in Y and in both directions.
The videos are saved in the folder results/MoveOnly, and the filenames indicate the parameters settings (ex: mag2_texturebg_moveY_amp0.125_alpha128.mp4) 

The video parameters, as in the beggining of the file movieOnly.py, are:
* amplitudes=[1/8, 1/2, 2]: Amplitude of movements of the non-magnified video, in pixels. Note that it is subpixel.
* target_mag=16: Amplitude of the movment of resulting magnified video
* num_frames=90: Number of frames to generate
* fps=30: Number of frames per second
* video_dur=3: Second duration of the video
* freqx=1.3: Frequence of the movement in the x axis
* freqy=1.5: Frequence of the movement in the y axis
* down=4: downsample from the original size.


##### Dataset 2: moveNdistort
Created by the file moveNdistort.py, it is composed by 27 videos (one for each foreground image) with motion amplification and other 27 corresponding to the videos with very slim movement. Contains periodic translation and distortion.

Each video has different setting, consisting in a combination of 3 factors:
* 3 different motion amplitudes;
* From the videos 2/3 has backgrounds from the background image data and 1/3 use a bland random color background; 
* Movement a random direction (either X or Y); distortion; combination of both movements.
The videos are saved in the folder results/MoveNdistort, and the filenames indicate the parameters settings (ex: mag14_texturebg_distort_amp0.5_alpha32.mp4)

The video parameters, as in the beggining of the file movieOnly.py, are:
* amplitudes=[1/8, 1/2, 2]: Amplitude of movements of the non-magnified video, in pixels. Note that it is subpixel.
* target_mag=16: Amplitude of the movment of resulting magnified video
* num_frames=90: Number of frames to generate
* fps=30: Number of frames per second
* video_dur=3: Second duration of the video
* freqx=1.3: Frequence of the movement in the x axis
* freqy=1.5: Frequence of the movement in the y axis
* freqdistort=1: Frequence of the distortion movement
* down=4: downsample from the original size.

##### TODO
The next step is to create another dataset for evaluating motion magnification in presence of large motion.
$$color(x)=f(x)*alpha(x)+bg(x)*(1-alpha(x))$$



