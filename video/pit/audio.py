import sys,wave,subprocess,numpy as np
from timeline import TL,END
VO,OUT=sys.argv[1],sys.argv[2]
SR=44100; N=int(END*SR); rng=np.random.default_rng(11); mix=np.zeros((N,2),np.float32)
def bp(x,lo,hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR)
    X*=1/(1+(lo/np.maximum(f,1))**4)/(1+(f/hi)**4); return np.fft.irfft(X,len(x)).astype(np.float32)
noise=lambda d: rng.standard_normal(int(d*SR)).astype(np.float32)
tt=lambda d: np.arange(int(d*SR))/SR
norm=lambda x: x/(np.abs(x).max()+1e-9)
fade=lambda t,d,a=.3: np.minimum(1,np.minimum(t,d-t)/a)
def put(x,t,g,pan=.5,echo=0):
    a=int(t*SR); x=np.asarray(x,np.float32)[:max(0,N-a)]
    mix[a:a+len(x),0]+=x*g*(1-pan)*1.41; mix[a:a+len(x),1]+=x*g*pan*1.41
    for k in range(1,echo+1): put(x,t+.19*k,g*.45**k,1-pan if k%2 else pan)
def whoosh(d=.7): t=tt(d); return norm(bp(noise(d),300,2200))*np.sin(np.pi*t/d)**3
def boom(d=1.8):
    t=tt(d); ph=2*np.pi*np.cumsum(32+60*np.exp(-t*6))/SR
    return norm(np.sin(ph)*np.exp(-t*3)+.4*norm(bp(noise(d),20,160))*np.exp(-t*8))
def grind(d):   # heavy stone platform moving
    t=tt(d); x=norm(bp(noise(d),25,220))*(.7+.3*np.sin(2*np.pi*7*t))+.25*norm(bp(noise(d),400,1400))*(.5+.5*np.sin(2*np.pi*3.1*t))
    return x*fade(t,d,.8)
def scrape(d=.32):  # knife on whetstone
    t=tt(d); return norm(bp(noise(d),2500,9000))*np.sin(np.pi*t/d)**2*(1+.5*np.sin(2*np.pi*90*t))
def hiss(d): t=tt(d); return norm(bp(noise(d),3000,12000))*fade(t,d,1.5)
def heart(d,bpm=74):
    t=tt(d); x=np.zeros(len(t),np.float32); p=60/bpm
    for o in (0,.28): u=(t-o)%p; x+=np.sin(2*np.pi*52*u)*np.exp(-u*18)*(1 if o==0 else .7)
    return x*fade(t,d,1.0)
def eerie(d):  # ghost
    t=tt(d); x=0
    for f in (523.3,554.4,784,830.6,1046.5):
        x=x+np.sin(2*np.pi*f*t+2*np.sin(2*np.pi*rng.uniform(.2,.5)*t))*(.6+.4*np.sin(2*np.pi*rng.uniform(.1,.3)*t))
    return norm(x)*fade(t,d,2.0)
def chime(d=3.5):
    t=tt(d); x=0
    for i,f in enumerate([1046.5,1318.5,1568,2093]):
        u=np.maximum(t-i*.2,0); x=x+np.sin(2*np.pi*f*u)*np.exp(-u*1.5)*(t>i*.2)
    return norm(x)
def drip():
    t=tt(.5); f=rng.uniform(1100,2200); return np.sin(2*np.pi*(f+400*np.exp(-t*40))*t)*np.exp(-t*14)
def pad(d,freqs,a=4.0):
    t=tt(d); x=0
    for f in freqs:
        for dt in (-.4,0,.5): x=x+np.sin(2*np.pi*(f+dt)*t+rng.uniform(0,6))*(.7+.3*np.sin(2*np.pi*rng.uniform(.05,.15)*t))
    return norm(x)*fade(t,d,a)
# beds: concrete room tone, low drone, distant drips
put(norm(bp(noise(END),40,300))[:N],0,.03)
put(pad(END-22,[49,73.4,98,116.5]),0,.07)
put(pad(END-306,[65.4,98,130.8,196,261.6],6),306,.085)       # resolve at the girl / ending
put(pad(20,[110,130.8,164.8]),220,.045)                      # grief on 202
for t in np.cumsum(rng.uniform(2.5,7,70)):
    if t<END-2: put(drip(),t,rng.uniform(.02,.05),rng.uniform(.15,.85),echo=3)
# cut whooshes (only on image changes)
for a,b in zip(TL,TL[1:]):
    if a[1]!=b[1]: put(whoosh(),b[0]-.3,.09,rng.uniform(.3,.7))
# level-number reveals
for t in (4.4,115.9,193.8,220.0,242.2,295.8): put(boom(),t,.32,echo=1)
for t in (26.7,53.7,77.3): put(boom(1.0),t,.16)
# knife sharpening
for t in np.arange(14.2,20.0,.75): put(scrape(),t,.07,.75,echo=1)
for t in np.arange(141.9,150.0,1.0): put(scrape(),t,.06,.7,echo=1)
put(scrape(.5),58.6,.09,.6,echo=2); put(scrape(.5),125.6,.10,.4,echo=2)
# platform
put(grind(5.0),27.2,.20); put(boom(1.2),32.0,.25)
put(grind(3.2),121.9,.18); put(boom(1.2),124.8,.22)
put(grind(3.0),152.3,.18); put(boom(1.2),155.0,.22)
put(grind(6.5),255.4,.16); put(grind(7.0),275.8,.14); put(grind(7.5),288.1,.18); put(boom(1.5),295.6,.3)
put(grind(16),314.0,.15)
# shout up the shaft / ignored
put(whoosh(1.0),82.2,.14,.5,echo=4)
# gas
put(hiss(7.0),104.5,.06)
# dread while tied up
put(heart(26,70),124.5,.2); put(heart(6,96),164.0,.22)
# the attack
for t in (164.3,165.4,166.5,168.6,170.2,171.6): put(boom(1.0),t,.3,rng.uniform(.3,.7)); put(whoosh(.35),t-.25,.18)
# ghost
put(eerie(11),182.6,.045); put(eerie(8),232.2,.045); put(eerie(7),320.8,.04)
# threat on 33
put(boom(1.2),216.2,.25)
# level 6 resolve + the fight down
put(boom(1.2),252.5,.2)
for t in (276.0,277.7,279.6,281.0,282.9,284.2,285.4): put(boom(.9),t,.28,rng.uniform(.25,.75)); put(whoosh(.3),t-.2,.15)
put(heart(12,88),275.8,.16)
# the message
put(chime(),268.5,.07); put(chime(),304.3,.08); put(chime(),307.0,.09); put(chime(4),326.8,.07)
put(boom(2.5),311.8,.2)
# voice + ducking
raw=subprocess.run(['ffmpeg','-v','error','-i',VO,'-f','f32le','-ac','2','-ar',str(SR),'-'],capture_output=True).stdout
v=np.frombuffer(raw,np.float32).reshape(-1,2)[:N]
env=np.abs(v).mean(1); k=int(.25*SR); env=np.convolve(env,np.ones(k)/k,'same')
mix*=np.pad(1-.45*np.clip(env/.05,0,1),(0,N-len(v)),constant_values=1)[:,None]
mix[:len(v)]+=v; mix=np.tanh(mix); mix*=np.minimum(1,(END-tt(END)[:N])/2.0)[:,None]
w=wave.open(OUT,'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
