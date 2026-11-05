import json, math, random
from collections import Counter

S=json.load(open('/tmp/eu_stats.json'))
# Saburtalo Tbilisi = Iberia 1999 (renommé) — identité partiellement vérifiée
S['Saburtalo Tbilisi']=S['Iberia 1999']
S['Lillestrøm']=S.get('Lillestrøm'); S['Viktoria Plzeň']=S.get('Viktoria Plzeň')

def stats(name):
    return S.get(name)

# moyenne de buts par équipe et par match dans chaque compétition
def baseline(lg):
    ts=[v for v in S.values() if v['lg']==lg and v['pld']>0]
    g=sum(v['gf'] for v in ts); n=sum(v['pld'] for v in ts)
    return g/n

MU={'EL':baseline('EL'),'CL':baseline('CL')}

def rate(team, lg):
    t=stats(team)
    mu=MU[lg]
    if not t or t['pld']==0: return None
    att=(t['gf']/t['pld'])/mu; de=(t['ga']/t['pld'])/mu
    # shrinkage vers la moyenne (échantillon 2-4 matchs)
    n=t['pld']; w=n/(n+4.0)          # shrinkage bayésien standard, k=4
    att=w*att+(1-w)*1.0; de=w*de+(1-w)*1.0
    return att,de,t['pld']

def lam(home,away,lg,home_adv=1.08):
    r1=rate(home,lg); r2=rate(away,lg)
    mu=MU[lg]
    if not r1 or not r2: return None
    l1=mu*r1[0]*r2[1]*home_adv
    l2=mu*r2[0]*r1[1]
    return max(0.15,min(l1,4.5)), max(0.15,min(l2,4.5))

def pois(l,K=9):
    v=[math.exp(-l)*l**k/math.factorial(k) for k in range(K+1)]; s=sum(v); return [x/s for x in v]

def grid_dc(l1,l2,rho=-0.10,K=9):
    h,a=pois(l1,K),pois(l2,K)
    g={}
    for i in range(K+1):
        for j in range(K+1):
            t=1.0
            if i==0 and j==0: t=1-l1*l2*rho
            elif i==0 and j==1: t=1+l1*rho
            elif i==1 and j==0: t=1+l2*rho
            elif i==1 and j==1: t=1-rho
            g[(i,j)]=h[i]*a[j]*max(t,0)
    s=sum(g.values()); return {k:v/s for k,v in g.items()}

def grid_bp(l1,l2,share=0.10,K=9):
    l3=share*min(l1,l2); a1=max(l1-l3,0.01); a2=max(l2-l3,0.01)
    g={}
    for i in range(K+1):
        for j in range(K+1):
            p=0.0
            for k in range(min(i,j)+1):
                p+=math.exp(-a1)*a1**(i-k)/math.factorial(i-k)*math.exp(-a2)*a2**(j-k)/math.factorial(j-k)*math.exp(-l3)*l3**k/math.factorial(k)
            g[(i,j)]=p
    s=sum(g.values()); return {k:v/s for k,v in g.items()}

def grid_nb(l1,l2,disp=8.0,K=9):
    def nb(l):
        v=[]
        for k in range(K+1):
            r=disp; p=r/(r+l)
            v.append(math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(p)+k*math.log(1-p)))
        s=sum(v); return [x/s for x in v]
    h,a=nb(l1),nb(l2); g={(i,j):h[i]*a[j] for i in range(K+1) for j in range(K+1)}
    s=sum(g.values()); return {k:v/s for k,v in g.items()}

def ensemble(l1,l2):
    A=grid_dc(l1,l2); B=grid_bp(l1,l2); C=grid_nb(l1,l2)
    g={k:0.45*A[k]+0.30*B[k]+0.25*C[k] for k in A}
    s=sum(g.values()); return {k:v/s for k,v in g.items()}

def analyse(home,away,lg,home_adv=1.08):
    r=lam(home,away,lg,home_adv)
    if not r: return None
    l1,l2=r; g=ensemble(l1,l2)
    top=sorted(g.items(),key=lambda x:-x[1])
    p1=sum(v for (i,j),v in g.items() if i>j); px=sum(v for (i,j),v in g.items() if i==j)
    btts=sum(v for (i,j),v in g.items() if i>0 and j>0)
    o25=sum(v for (i,j),v in g.items() if i+j>2.5)
    # Monte Carlo + robustesse
    rng=random.Random(7); c=Counter(); leaders=[]
    for _ in range(200000):
        s1=l1*rng.uniform(0.85,1.15); s2=l2*rng.uniform(0.85,1.15)
        i=0; L=math.exp(-s1); p=1.0
        while p>L: i+=1; p*=rng.random()
        j=0; L=math.exp(-s2); p=1.0
        while p>L: j+=1; p*=rng.random()
        if i<=8 and j<=8: c[(i-1,j-1)]+=1
    tot=sum(c.values())
    mc={k:v/tot for k,v in c.items()}
    best=top[0][0]
    stab=sum(1 for _ in range(400))
    st=random.Random(11); same=0
    for _ in range(400):
        gg=ensemble(l1*st.uniform(0.85,1.15), l2*st.uniform(0.85,1.15))
        if max(gg,key=gg.get)==best: same+=1
    return {
     'lambda':[round(l1,3),round(l2,3)],
     'p1':round(p1*100,1),'px':round(px*100,1),'p2':round(100*(1-p1-px),1),
     'btts':round(btts*100,1),'o25':round(o25*100,1),
     'top':[{'s':f"{i}-{j}",'p':round(v*100,2)} for (i,j),v in top[:10]],
     'top3':round(sum(v for _,v in top[:3])*100,1),
     'mc_top':{f"{i}-{j}":round(v*100,2) for (i,j),v in sorted(mc.items(),key=lambda x:-x[1])[:3]},
     'stability':round(100*same/400,1),
    }

FIX=[("17:45","Trabzonspor","SC Freiburg","CL"),("17:45","KuPS","Gent","CL"),
 ("17:45","Atalanta","Kairat","CL"),("17:45","Real Sociedad","Lyon","EL"),
 ("17:45","Lillestrøm","Viktoria Plzeň","CL"),("17:45","Milan","Ferencváros","EL"),
 ("20:00","Panathinaikos","CSKA Sofia","CL"),("20:00","Getafe","Brighton & Hove Albion","EL"),
 ("20:00","Twente","Pafos","CL"),("20:00","Lugano","Kauno Žalgiris","CL"),
 ("20:00","Copenhagen","Saburtalo Tbilisi","CL"),("20:00","Sunderland","Dinamo Zagreb","EL"),
 ("20:00","Benfica","Lech Poznań","EL"),("20:00","Sturm Graz","Celje","CL"),
 ("20:00","AZ","Juventus","EL")]
res=[]
for t,h,a,lg in FIX:
    r=analyse(h,a,lg)
    res.append({'time':t,'home':h,'away':a,'comp':lg,'data':r})
json.dump({'mu':MU,'matches':res},open('data.json','w'),ensure_ascii=False,indent=1)
for m in res:
    d=m['data']
    if not d: print(m['home'],'NO DATA'); continue
    print("%-14s %-24s %s | 1 %.0f X %.0f 2 %.0f | top %s %.1f%% | stab %.0f%%"%(
      m['time'], m['home']+" - "+m['away'], d['lambda'], d['p1'],d['px'],d['p2'], d['top'][0]['s'], d['top'][0]['p'], d['stability']))
