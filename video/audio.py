import sys,wave,subprocess,numpy as np
from timeline import TL,END
VO,OUT=sys.argv[1],sys.argv[2]
SR=44100; N=int(END*SR); rng=np.random.default_rng(7)
mix=np.zeros((N,2),np.float32)
def bp(x,lo,hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR)
    X*=1/(1+(lo/np.maximum(f,1))**4)/(1+(f/hi)**4); return np.fft.irfft(X,len(x)).astype(np.float32)
def noise(d): return rng.standard_normal(int(d*SR)).astype(np.float32)
def tt(d): return np.arange(int(d*SR))/SR
def norm(x): return x/ (np.abs(x).max()+1e-9)
def put(x,t,g,pan=0.5):
    a=int(t*SR); x=x[:max(0,N-a)]
    mix[a:a+len(x),0]+=x*g*(1-pan)*2**.5; mix[a:a+len(x),1]+=x*g*pan*2**.5
def whoosh(d=.7):
    t=tt(d); return norm(bp(noise(d),400,2500))*np.sin(np.pi*t/d)**3
def boom(d=1.8):
    t=tt(d); ph=2*np.pi*np.cumsum(35+60*np.exp(-t*6))/SR
    return norm(np.sin(ph)*np.exp(-t*3)+.5*bp(noise(d),20,180)*np.exp(-t*8)/.05*.05)
def glass(d=1.2):
    t=tt(d); x=bp(noise(d),3000,12000)*np.exp(-t*7)*3
    for _ in range(14):
        o=rng.uniform(0,.35); f=rng.uniform(2500,9000); tt2=np.maximum(t-o,0)
        x+=np.sin(2*np.pi*f*tt2)*np.exp(-tt2*rng.uniform(8,25))*(t>o)*.3
    return norm(x)
def growl(d=1.8):
    t=tt(d); f=75+25*np.sin(2*np.pi*.7*t+rng.uniform(0,6)); ph=2*np.pi*np.cumsum(f)/SR
    x=sum(np.sin(k*ph)/k for k in range(1,7))*(.6+.4*np.sign(np.sin(2*np.pi*31*t)))
    x=x+2*bp(noise(d),80,500)*norm(np.abs(x))
    return norm(bp(x.astype(np.float32),60,900))*np.sin(np.pi*t/d)**1.5
def alarm(d):
    t=tt(d); return np.tanh(3*np.sin(2*np.pi*(760+120*(np.floor(t*2)%2))*t))*.5*(np.sin(2*np.pi*2*t)>-0.3)
def static(d):
    t=tt(d); x=norm(bp(noise(d),600,3500))*(.6+.4*np.sin(2*np.pi*.4*t))
    x+=.15*np.sin(2*np.pi*(1200+400*np.sin(2*np.pi*.23*t))*t)
    return x*np.minimum(1,np.minimum(t,d-t)/.4)
def heli(d):
    t=tt(d); return norm(bp(noise(d),40,500))*(.35+.65*(np.sin(2*np.pi*11*t)>0.2))*np.sin(np.pi*t/d)
def heart(d,bpm=72):
    t=tt(d); x=np.zeros(len(t),np.float32); p=60/bpm
    for o in (0,.28):
        u=(t-o)%p; x+=np.sin(2*np.pi*52*u)*np.exp(-u*18)*(1 if o==0 else .7)
    return x*np.minimum(1,np.minimum(t,d-t)/1.0)
def shimmer(d=3.5):
    t=tt(d); x=0
    for i,f in enumerate([1046.5,1318.5,1568,2093,2637]):
        u=np.maximum(t-i*.18,0); x=x+np.sin(2*np.pi*f*u)*np.exp(-u*1.6)*(t>i*.18)
    return norm(x)
def pad(d,freqs):
    t=tt(d); x=0
    for f in freqs:
        for dt in (-.4,0,.5):
            x=x+np.sin(2*np.pi*(f+dt)*t+rng.uniform(0,6))*(.7+.3*np.sin(2*np.pi*rng.uniform(.05,.15)*t))
    return norm(x)*np.minimum(1,np.minimum(t,d-t)/4.0)
# beds
wind=norm(bp(noise(END),150,900))*(.55+.45*np.sin(2*np.pi*.07*tt(END))*np.sin(2*np.pi*.031*tt(END)))
put(wind[:N],0,.035)
put(pad(296,[55,82.4,110,130.8]),0,.075)               # dark drone
put(pad(18,[110,164.8,220,261.6]),197,.05)             # sorrow layer
put(pad(END-290,[65.4,98,130.8,164.8,196,261.6]),290,.09)  # hopeful resolve
# cuts
for i,c in enumerate(TL[1:]): put(whoosh(),c[0]-.3,.10,.35+.3*(i%2))
G=lambda t,g=.16,p=.5: put(growl(),t,g,p)
for t,p in [(10.8,.3),(13,.7),(33.4,.4),(35.5,.65),(81.4,.3),(83.5,.7),(121.8,.4),(124.5,.65),(127,.3),(129.5,.6),
            (156.5,.7),(158.8,.3),(162,.5),(242.2,.3),(244.5,.6),(247.5,.4),(251.5,.3),(253.5,.6),(256.5,.4),
            (270.3,.3),(272,.7),(274.5,.3),(283.2,.4),(285,.6),(287,.4)]: G(t,.15,p)
put(alarm(8.5),28.6,.045); put(glass(),28.5,.16,.35); put(boom(),28.4,.3)
put(static(7.0),89.9,.05,.4)
put(heli(8.5),103.5,.14,.7)
put(whoosh(.4),150.9,.25); put(boom(),151.2,.32)
put(heart(10),156.2,.22); put(heart(22,84),270,.22)
put(boom(),186.0,.3); put(boom(2.5),199.0,.2)
for t in (250.3,251.6,252.7,254.2,255.0,257.4): put(boom(1.0),t,.3,rng.uniform(.3,.7))
put(glass(),250.4,.2,.3); put(glass(),283.0,.16,.4)
put(shimmer(),260.2,.10); put(shimmer(),301.2,.08); put(shimmer(),305.0,.07)
put(boom(3.0),289.6,.45); put(whoosh(1.4),288.4,.2)
# voice
raw=subprocess.run(['ffmpeg','-v','error','-i',VO,'-f','f32le','-ac','2','-ar',str(SR),'-'],capture_output=True).stdout
v=np.frombuffer(raw,np.float32).reshape(-1,2)[:N]
env=np.abs(v).mean(1); k=int(.25*SR); env=np.convolve(env,np.ones(k)/k,'same')
duck=1-.45*np.clip(env/.05,0,1)                      # duck effects under speech
mix*=np.pad(duck,(0,N-len(duck)),constant_values=1)[:,None]
mix[:len(v)]+=v
mix=np.tanh(mix*1.0)
f=np.minimum(1,(END-tt(END)[:N])/1.5)[:,None]; mix*=f
w=wave.open(OUT,'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
