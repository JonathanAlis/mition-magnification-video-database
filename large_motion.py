import GenerateMagVideo as magv
import random
import numpy as np
import cv2
from moviepy.editor import VideoClip

#params:
reproductive=True
down=4 #downsize of the final video compared to original change at our will
dir_name='moveHalfLarge_downsize_'+str(down)
target_mag=64/down #amplitude of the resulting magnified video in pixels
amplitudes=[target_mag/16, target_mag/8, target_mag/4] #amplitude of movements of nonmag
num_frames=90 #number of frames to generate
fps=30 #number of frames per second
video_dur=3 #second duration of the video
freqx_range=[1.5,2.5] #frequence of the movement in the x axis
freqy_range=[1.5, 2.5] #frequence of the movement in the 7 axis
freqdistort=1 #frequence of the distortion movent
toCrop=False


def testImg(im):
    if len(im)==2:
        testimg = np.zeros((im[0],im[1],3),np.uint8)
        testam = np.zeros((im[0],im[1],3),np.uint8)
    else:
        testimg = np.zeros(im.shape,np.np.uint8)
        testam = np.zeros(im.shape,np.uint8)
    midx=testimg.shape[0]//2
    midy=testimg.shape[1]//2
    testimg[midx-50:midx+50,midy-50:midy+50,:]=255
    testam[midx-50:midx+50,midy-50:midy+50,:]=255
    return testimg,testam



def to_0255(im):
    im=im*255
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return im


from os import listdir, makedirs
from os.path import isfile, join, exists
if not exists('resultVideos/'+dir_name+'/'):
    makedirs('resultVideos/'+dir_name+'/')

alphamates_dir='./gt_training_highres'
im_dir='./input_training_highres'
bg_dir='./backgrounds'
am_files = [alphamates_dir+'/'+f for f in listdir(alphamates_dir) if isfile(join(alphamates_dir, f))]
im_files = [im_dir+'/'+f for f in listdir(im_dir) if isfile(join(im_dir, f))]
bg_files = [bg_dir+'/'+f for f in listdir(bg_dir) if isfile(join(bg_dir, f))]


#cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)


combinations=[[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0],[0,2,0],[1,2,0],[2,2,0],
              [0,0,1],[1,0,1],[2,0,1],[0,1,1],[1,1,1],[2,1,1],[0,2,1],[1,2,1],[2,2,1],
              [0,0,2],[1,0,2],[2,0,2],[0,1,2],[1,1,2],[2,1,2],[0,2,2],[1,2,2],[2,2,2]]
for i in range(len(am_files)):
    magname='resultVideos/'+dir_name+'/mag'+str(i)+'_'
    nomagname='resultVideos/'+dir_name+'/orig'+str(i)+'_'
    if combinations[i][0]==0:
        v=magv.GenerateMagVideo(im_files[i],am_files[i],bg_color=(random.random(),random.random(),random.random()))
        magname+='plainbg_'
        nomagname+='plainbg_'        
    elif combinations[i][0]==1:
        random.shuffle(bg_files)
        v=magv.GenerateMagVideo(im_files[i],am_files[i],bg_files.pop())
        magname+='texturebg_'
        nomagname+='texturebg_'
    else:
        random.shuffle(bg_files)
        v=magv.GenerateMagVideo(im_files[i],am_files[i],bg_files.pop(),BGoffsetX=10)
        magname+='texturemovebg_'
        nomagname+='texturemovebg_'
    #v.downsize(20,20)

    move=[0,0]    
    distort=[0,0]
    if combinations[i][2]==2:
        magname+='moveXY_'
        nomagname+='moveXY_'
        move[0]=1
        move[1]=1
    if combinations[i][2]==1:
        move[0]=1
        magname+='moveX_'
        nomagname+='moveX_'
    if combinations[i][2]==0:
        move[1]=1
        magname+='moveY_'
        nomagname+='moveY_'


    amplitude=amplitudes[combinations[i][1]]  #2**(2*combinations[i][1]-3)#(2**[-3,-1,1])
    #1/8, 1/2, 2
    alpha=target_mag/amplitude
    nomagname+='amp'+str(amplitude)
    magname+='amp'+str(amplitude)+'_'    
    magname+='alpha'+str(int(alpha))
    #distort=[0,0]#comentar depois
    magname+='_Large.mp4'
    nomagname+='_Large.mp4'
    
    import os
    print('video:',magname)
    if os.path.isfile(magname) and os.path.isfile(nomagname):
        print('already exists, skipping')
        continue
    if reproductive:
        random.seed(i)
    v.start_video(num_frames,
                  move[0],move[1],
                  random.uniform(freqx_range[0],freqx_range[1]),random.uniform(freqy_range[0],freqy_range[1]),
                  distort[0],distort[1],freqdistort,
                  alpha=alpha,fps=fps,pixel_mag=target_mag,downscale=down,addLarge=True,moveBG=combinations[i][0]==2,
                  magname=magname,nomagname=nomagname,toCrop=toCrop)    
    
    def get_frame(v,t,magf=True,view=True):
        if magf:
            frame=v.mframe_at_t(t,view=view)
        else:
            frame=v.frame_at_t(t,view=view)
        frame=to_0255(frame)
        return frame 

    nomag = lambda t : get_frame(v,t,magf=False) 
    mag = lambda t : get_frame(v,t,magf=True)
    clip = VideoClip(mag, duration=3) # seconds
    #clip.write_videofile(v.magname, fps=v.fps, codec='rawvideo',preset="placebo")
    clip.write_videofile(v.magname, fps=v.fps, codec='libx264',ffmpeg_params=['-preset', 'veryfast', '-qp', '0'])

    clip = VideoClip(nomag, duration=3) # 2 seconds
    clip.write_videofile(v.nomagname, fps=v.fps, codec='libx264',ffmpeg_params=['-preset', 'veryfast', '-qp', '0'])
    