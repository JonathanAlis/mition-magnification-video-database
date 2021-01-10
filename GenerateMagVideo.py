import cv2
import numpy as np
import math
import time
import random
from scipy.optimize import fmin
import math
from moviepy.editor import VideoClip


def optimize_display(img_list):
    from screeninfo import get_monitors
    for m in get_monitors():
        w=m.width-200
        h=m.height-200

    full=cv2.hconcat(img_list)
    oldw=full.shape[0]
    oldh=full.shape[1]
    if oldw>w or oldh>h:
        scale=min(w/oldh,h/oldw)
        neww=oldw*scale
        newh=oldh*scale
        full = cv2.resize(full,(int(newh),int(neww)))
    return full


class GenerateMagVideo:
    def __init__(self,im_name,am_name,bg_name='',bg_color=(0.0,1.0,0.0)):
        try:
            self.im=cv2.imread(im_name)*1.0/255
        except:
            self.im=im_name

        try:
            self.am=cv2.imread(am_name)*1.0/255
        except:
            self.am=am_name

        
        self.shp=self.am.shape[:2]
        if bg_name=='':
            ones=np.ones(self.am.shape,self.am.dtype)
            self.fullbg=ones.copy()
            self.fullbg[:,:,:]=bg_color
        else:
            self.fullbg=cv2.imread(bg_name)*1.0/255
        self.bg_crop()

    def bg_crop(self):
        bgsize=self.fullbg.shape[:2]
        xr=self.shp[0]/bgsize[0]
        yr=self.shp[1]/bgsize[1]
        
        if xr>1 or yr>1:
            #linha pelo maior

            self.fullbg=cv2.resize(self.fullbg,(int(max(xr,yr)*bgsize[1]),int(max(xr,yr)*bgsize[0])))    
        else:
            self.fullbg=cv2.resize(self.fullbg,(int(max(xr,yr)*bgsize[1]),int(max(xr,yr)*bgsize[0])))
        bgsz=self.fullbg.shape[:2]
        diff=(bgsz[0]-self.shp[0],bgsz[1]-self.shp[1])
        self.fullbg=self.fullbg[diff[0]//2:bgsz[0]-diff[0]//2,diff[1]//2:bgsz[1]-diff[1]//2]
        self.fullbg=self.fullbg[:self.shp[0],:self.shp[1]]
                
        

    def downsize(self,factor, multiple_of=4):
        self.am=cv2.resize(self.am,(self.shp[1]//factor,self.shp[0]//factor))#depois comenta isso
        self.im=cv2.resize(self.im,(self.shp[1]//factor,self.shp[0]//factor))
        self.fullbg=cv2.resize(self.fullbg,(self.shp[1]//factor,self.shp[0]//factor))
        self.shp=(self.am.shape[0],self.am.shape[1])
        #self.shp=(self.shp[0]//multiple_of*multiple_of,self.shp[1]//multiple_of*multiple_of)
        #self.am=self.am[0:self.shp[0],0:self.shp[1],:]
        #self.im=self.im[0:self.shp[0],0:self.shp[1],:]
        #self.fullbg=self.fullbg[0:self.shp[0],0:self.shp[1],:]
        
    def start_video(self,num_frames,movex,movey,fmx,fmy,distx,disty,fd,alpha=10,fps=30,pixel_mag=30,downscale=1,magname='large.mp4',nomagname='short.mp4',toCrop=True):
        self.target_mag=pixel_mag*downscale
        self.alpha=alpha
        self.movex=movex*self.target_mag/self.alpha
        self.movey=movey*self.target_mag/self.alpha
        self.fmx=fmx
        self.fmy=fmy
        self.fd=fd
        self.distx=distx*self.target_mag/self.alpha
        self.disty=disty*self.target_mag/self.alpha
        self.num_frames=num_frames
        self.fps=fps
        self.movephasex= random.random()*np.pi 
        self.movephasey= random.random()*np.pi 
        self.distphase= random.random()*np.pi 
        self.yrange=[0,self.shp[0]]
        self.xrange=[0,self.shp[1]]
        self.downscale=downscale
        self.current_frame_num=0       
        self.magname=magname
        self.nomagname=nomagname
        self.toCrop=toCrop
        
        self.vidsize=self.shp
        self.static_map()
        self.center()
        self.crop_values()
        print('mag pixel displacement:',self.target_mag/downscale,',in HR',self.target_mag)
        print('no mag pixel displacement:',self.target_mag/downscale/self.alpha,',in HR',self.target_mag/self.alpha)
        print('alpha:',self.alpha)
        

    def static_map(self):
        self.map_x = np.zeros(self.shp,np.float32)
        self.map_y = np.zeros(self.shp,np.float32)
        for i in range(0,self.shp[0]):
            for j in range(0,self.shp[1]):#
                self.map_x[i,j] = j
                self.map_y[i,j] = i 


    def crop_values(self):
        fx=lambda x: self.movex*self.alpha*np.sin(2*np.pi*self.fmx*x+self.movephasex)+self.distx*self.alpha*(np.sin(2*np.pi*self.fd*x+self.distphase))
        fy=lambda y: self.movey*self.alpha*np.sin(2*np.pi*self.fmy*y+self.movephasey)+self.disty*self.alpha*(np.sin(2*np.pi*self.fd*y+self.distphase))
        min_x = fx(fmin(fx, 0))
        max_x = -fx(fmin(lambda x: -fx(x), 0))
        x_abs = max(abs(min_x),abs(max_x))
        min_y = fy(fmin(fy, 0))
        max_y = -fy(fmin(lambda x: -fy(x), 0))
        print(min_y)
        print(max_y)
        y_abs = max(abs(min_y),abs(max_y))
        self.xrange=(math.ceil(x_abs),math.floor(self.shp[1]-x_abs))
        self.yrange=(math.ceil(y_abs),math.floor(self.shp[0]-y_abs))
        print('xrange',[0,self.shp[1]])
        print('yrange',[0,self.shp[0]])
        print('cropx into',self.xrange)
        print('cropy into',self.yrange)

    def moving_maps(self,frame_num,ampx,ampy):  
        x=ampx*np.sin(2*np.pi*self.fmx/self.fps*frame_num+self.movephasex)
        y=ampy*np.sin(2*np.pi*self.fmy/self.fps*frame_num+self.movephasey)        
        vecs_x=np.ones(self.shp,np.float32)*x
        vecs_y=np.ones(self.shp,np.float32)*y
        
        return (vecs_x,vecs_y)

    def center(self):
        mmts= cv2.moments(cv2.cvtColor(self.am.astype(np.float32), cv2.COLOR_BGR2GRAY))
        self.center=(mmts['m10']/mmts['m00'],mmts['m01']/mmts['m00'])
        self.radius=min([self.shp[0]-self.center[0],self.center[0],self.shp[1]-self.center[1],self.center[1]])

    def distorting_maps(self,frame_num,ampx,ampy):
        vecs_x=np.zeros(self.shp,np.float32)
        vecs_y=np.zeros(self.shp,np.float32)
        if ampx!=0 or ampy!=0:
            for i in range(self.shp[0]):
                for j in range(self.shp[1]):
                    distx=np.sqrt((j-self.center[0])*(j-self.center[0]))
                    disty=np.sqrt((i-self.center[1])*(i-self.center[1]))
                    x=distx/self.radius*ampx*(np.sin(2*np.pi*self.fd/self.fps*frame_num+self.distphase))
                    y=disty/self.radius*ampy*(-np.sin(2*np.pi*self.fd/self.fps*frame_num+self.distphase))
                    vecs_x[i,j]=x*(j-self.center[0])/distx
                    vecs_y[i,j]=y*(i-self.center[1])/disty
        #print(vecs_x)
        return (vecs_x,vecs_y)
    

    def remap_n_alphamate(self,vecsx,vecsy):
        #print(self.map_x)
        
        amr=cv2.remap( self.am, self.map_x+vecsx,self.map_y+vecsy, cv2.INTER_CUBIC) #alpha mate remaped
        imr=cv2.remap( self.im, self.map_x+vecsx,self.map_y+vecsy, cv2.INTER_CUBIC) #image remaped       
        amimr=cv2.multiply(amr,imr) #alpha mate image remaped
        newbg=self.fullbg*(1-amr) #background without the image
        frame=amimr+newbg        
        return frame

    def mframe_at_t(self,t,view=False):
        frame_num=t*self.fps
        mx,my=self.moving_maps(frame_num,self.movex*self.alpha,self.movey*self.alpha)        
        dx,dy=self.distorting_maps(frame_num,self.distx*self.alpha,self.disty*self.alpha)
        mframe=self.remap_n_alphamate(dx+mx,my+dy)
        #mframe=mframe[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:] 
        #mframe=self.crop_and_down(self,mframe)
        if view:
            withrect=mframe.copy()
            cv2.rectangle(withrect, (self.xrange[0],self.yrange[0]), (self.xrange[1],self.yrange[1]),(255,255,255),10)
            view=optimize_display([withrect,mframe])            
            cv2.imshow('magnified',view)
            #cropview=optimize_display([self.crop_and_down(mframe)])
            #cv2.imshow('crop',cropview)
            key=cv2.waitKey(int(1000/self.fps)) & 0xFF  
        mframe=self.crop_and_down(mframe)    
        return mframe

    def frame_at_t(self,t,view=False):
        frame_num=t*self.fps
        mx,my=self.moving_maps(frame_num,self.movex,self.movey)        
        dx,dy=self.distorting_maps(frame_num,self.distx,self.disty)
        frame=self.remap_n_alphamate(mx+dx,my+dy)
        #frame=self.crop_and_down(self,frame)
        if view:
            withrect=frame.copy()
            cv2.rectangle(withrect, (self.xrange[0],self.yrange[0]), (self.xrange[1],self.yrange[1]),(255,255,0),4)
            view=optimize_display([frame,withrect])            
            cv2.imshow('no magnification',view)
            #cropview=optimize_display([self.crop_and_down(frame)])
            #cv2.imshow('crop',frame)
            key=cv2.waitKey(int(1000/self.fps)) & 0xFF    
        frame=self.crop_and_down(frame)
        return frame

    def crop_and_down(self,frame,nods=False):
        if self.toCrop:
            frame=frame[self.yrange[0]:self.yrange[1],self.xrange[0]:self.xrange[1],:]
        frame=cv2.resize(frame,(frame.shape[1]//self.downscale,frame.shape[0]//self.downscale))

        #if not nods:
        #    print(frame.shape)
        #    frame=cv2.resize(frame,(frame.shape[1]//self.downscale,frame.shape[0]//self.downscale))
        return frame

    def write(self,framelarge,frameshort):
        self.large.write(framelarge)
        self.short.write(frameshort)
