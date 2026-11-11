import sys,os,math,subprocess,numpy as np
from PIL import Image
from timeline import TL,END
IMG,VO,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
W,H,FPS,XF=1920,1080,30,0.45
src={}
def load(n):
    if n not in src:
        p=[f for f in os.listdir(IMG) if f.split('.')[0]==str(n)][0]
        im=Image.open(os.path.join(IMG,p)).convert('RGB')
        # cover-crop to 16:9 then upscale for clean zooms
        w,h=im.size; tw=min(w,h*16/9); th=tw*9/16
        im=im.crop((int((w-tw)/2),int((h-th)/2),int((w+tw)/2),int((h+th)/2))).resize((2880,1620),Image.LANCZOS)
        src[n]=im
    return src[n]
rng=np.random.default_rng(3)
def frame(i,t):
    s,n,z0,z1,f0,f1,sh=TL[i]
    e=TL[i+1][0] if i+1<len(TL) else END
    u=min(max((t-s)/(e+XF-s),0),1); u=0.5-0.5*math.cos(math.pi*u)
    if sh: # quick punch-in at the start of impact shots
        k=min((t-s)/0.35,1); u=max(u,0.25*(1-(1-k)**3))
    z=z0+(z1-z0)*u; fx=f0[0]+(f1[0]-f0[0])*u; fy=f0[1]+(f1[1]-f0[1])*u
    im=load(n); SW,SH=im.size; cw,ch=SW/z,SH/z
    x=min(max(fx*SW-cw/2,0),SW-cw); y=min(max(fy*SH-ch/2,0),SH-ch)
    if sh and t-s<0.9:
        a=14*(1-(t-s)/0.9); x=min(max(x+rng.uniform(-a,a),0),SW-cw); y=min(max(y+rng.uniform(-a,a),0),SH-ch)
    return im.resize((W,H),Image.BILINEAR,box=(x,y,x+cw,y+ch))
ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
 '-i',VO,'-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k',
 '-movflags','+faststart','-shortest',OUT],stdin=subprocess.PIPE)
N=int(END*FPS); i=0
for k in range(N):
    t=k/FPS
    while i+1<len(TL) and t>=TL[i+1][0]: i+=1
    f=frame(i,t)
    if i+1<len(TL) and TL[i+1][0]-t<XF/2*0+XF and False: pass
    if i>0 and t-TL[i][0]<XF:   # crossfade from previous shot
        f=Image.blend(frame(i-1,t),f,(t-TL[i][0])/XF)
    a=np.asarray(f)
    g=min(t/0.8,(END-t)/1.2,1)
    if g<1: a=(a*max(g,0)).astype(np.uint8)
    ff.stdin.write(a.tobytes())
    if k%900==0: print(k,N,flush=True)
ff.stdin.close(); ff.wait()
