import sys,os,json,math,subprocess,numpy as np,cv2
from timeline import TL,END,ANIM
IMG,AUD,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
T0=float(sys.argv[4]) if len(sys.argv)>4 else 0; T1=float(sys.argv[5]) if len(sys.argv)>5 else END
FILES=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'images.json')))
W,H,FPS,XF=1920,1080,30,0.45; SW,SH=2400,1350
rng=np.random.default_rng(5); src={}; masks={}
def emask(cx,cy,rx,ry):
    x0,x1=max(int((cx-rx*1.5)*SW),0),min(int((cx+rx*1.5)*SW),SW); y0,y1=max(int((cy-ry*1.5)*SH),0),min(int((cy+ry*1.5)*SH),SH)
    yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
    d=np.sqrt(((xx/SW-cx)/(rx*1.5))**2+((yy/SH-cy)/(ry*1.5))**2); m=np.clip(1-d,0,1); m=m*m*(3-2*m)
    return (x0,x1,y0,y1,xx,yy,m.astype(np.float32))
def load(n):
    if n not in src:
        im=cv2.imread(os.path.join(IMG,FILES[n])); h,w=im.shape[:2]; tw=int(min(w,h*16/9)); th=int(tw*9/16)
        im=im[(h-th)//2:(h-th)//2+th,(w-tw)//2:(w-tw)//2+tw]; src[n]=cv2.resize(im,(SW,SH),interpolation=cv2.INTER_LANCZOS4)
        a=ANIM.get(n,{}); masks[n]={k:[emask(*r[:4]) for r in a.get(k,[])] for k in ('bob','glow','drift')}
    return src[n]
# fog + dust textures (output space)
fogtex=cv2.GaussianBlur(rng.random((270,960)).astype(np.float32),(0,0),22); fogtex=(fogtex-fogtex.min())/(np.ptp(fogtex))
fogtex=np.concatenate([fogtex,fogtex[:,::-1]],1)
vgrad=np.clip((np.linspace(0,1,270)-.35)/.5,0,1)[:,None].astype(np.float32)
P=rng.random((160,5)).astype(np.float32)  # x,y,size,speed,phase
def animate(n,t,u):
    im=load(n); a=ANIM.get(n,{}); out=im
    if a.get('bob') or a.get('drift') or a.get('glow'): out=im.copy()
    for r,(x0,x1,y0,y1,xx,yy,m) in zip(a.get('drift',[]),masks[n]['drift']):
        out[y0:y1,x0:x1]=cv2.remap(im,xx-r[4]*u*m,yy-r[5]*u*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    for r,(x0,x1,y0,y1,xx,yy,m) in zip(a.get('bob',[]),masks[n]['bob']):
        s=math.sin(2*math.pi*t/r[6]+r[0]*9)
        out[y0:y1,x0:x1]=cv2.remap(out if a.get('drift') else im,xx-r[4]*s*m,yy-r[5]*s*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    for r,(x0,x1,y0,y1,xx,yy,m) in zip(a.get('glow',[]),masks[n]['glow']):
        s=.5+.5*math.sin(2*math.pi*t/r[5])
        c=np.array(r[4][::-1],np.float32)*s
        out[y0:y1,x0:x1]=np.clip(out[y0:y1,x0:x1].astype(np.float32)+m[...,None]*c,0,255).astype(np.uint8)
    return out
def overlay(f,n,t):
    a=ANIM.get(n,{})
    if a.get('fog'):
        col,al=a['fog']; o=int(t*14)%960; o2=int(t*23+300)%960
        g=(fogtex[:,o:o+480]*fogtex[::-1,o2:o2+480])[:, :480]; g=cv2.resize(np.clip(g*3.2,0,1)*vgrad*al,(W,H))[...,None]
        f=(f*(1-g)+np.array(col[::-1],np.float32)*g)
    d=a.get('dust',0)
    if d:
        buf=np.zeros((270,480),np.float32); k=int(60*min(d,2.5))
        fall=a.get('fall',0)
        x=(P[:k,0]+.012*np.sin(t*.4+P[:k,4]*6)+t*.004*(P[:k,3]-.5))%1
        y=(P[:k,1]-t*(.012+.02*P[:k,3])*(1 if not fall else -18))%1
        tw=.5+.5*np.sin(t*1.3+P[:k,4]*20)
        np.add.at(buf,((y*269).astype(int),(x*479).astype(int)),(.5+P[:k,2])*tw)
        buf=cv2.GaussianBlur(buf,(0,0),1.1 if not fall else 0.8)
        if fall: buf=cv2.blur(buf,(1,25))*6
        f=np.asarray(f,np.float32)+cv2.resize(buf,(W,H))[...,None]*(150 if not fall else 110)
    return f
def frame(i,t):
    s,n,z0,z1,f0,f1,sh=TL[i]; e=TL[i+1][0] if i+1<len(TL) else END
    u=min(max((t-s)/(e+XF-s),0),1); ul=u; u=0.5-0.5*math.cos(math.pi*u)
    if sh: k=min((t-s)/0.35,1); u=max(u,0.22*(1-(1-k)**3))
    z=z0+(z1-z0)*u; fx=f0[0]+(f1[0]-f0[0])*u; fy=f0[1]+(f1[1]-f0[1])*u
    im=animate(n,t,ul); cw,ch=SW/z,SH/z
    x=fx*SW-cw/2; y=fy*SH-ch/2
    # handheld drift so no shot is ever perfectly still
    x+=6*math.sin(t*.53+i); y+=4*math.sin(t*.41+i*2)
    v=ANIM.get(n,{}).get('vib',0)
    if v: x+=rng.uniform(-v,v); y+=rng.uniform(-v,v)
    if sh and t-s<0.9: a=16*(1-(t-s)/0.9); x+=rng.uniform(-a,a); y+=rng.uniform(-a,a)
    x=min(max(x,0),SW-cw); y=min(max(y,0),SH-ch); sc=W/cw
    f=cv2.warpAffine(im,np.float32([[sc,0,-x*sc],[0,sc,-y*sc]]),(W,H),flags=cv2.INTER_LINEAR)
    return overlay(f,n,t).astype(np.float32)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32); vig=(1-.38*np.clip(((xx/W-.5)**2+(yy/H-.5)**2)*2.2,0,1))[...,None]
ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-']+
 (['-ss',str(T0),'-t',str(T1-T0),'-i',AUD] if AUD!='none' else [])+['-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',OUT],stdin=subprocess.PIPE)
i=0
for k in range(int(T0*FPS),int(T1*FPS)):
    t=k/FPS
    while i+1<len(TL) and t>=TL[i+1][0]: i+=1
    f=frame(i,t)
    if i>0 and t-TL[i][0]<XF and TL[i-1][1]!=TL[i][1]:
        b=(t-TL[i][0])/XF; f=frame(i-1,t)*(1-b)+f*b
    elif i>0 and t-TL[i][0]<0.2: # same image: quick dissolve between framings
        b=(t-TL[i][0])/0.2; f=frame(i-1,t)*(1-b)+f*b
    f*=vig*(1+.02*math.sin(t*7.3)*math.sin(t*2.9))*max(min(t/0.8,(END-t)/1.5,1),0)
    ff.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
    if k%900==0: print(k,flush=True)
ff.stdin.close(); ff.wait()
