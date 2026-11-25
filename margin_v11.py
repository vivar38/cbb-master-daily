"""Enhanced pregame margin features. No odds, no future or prior-season team data."""
from dataclasses import dataclass,field
import numpy as np
import pandas as pd
from cbb_model import PRIOR
KEYS=['off','def','pace','efg','tov','orb','ftr','deftov','deforb','defefg','defftr','sos','opponent_n']
DEFAULT=PRIOR|dict(deftov=.18,deforb=.30,defefg=.5,defftr=.33,sos=0.,opponent_n=0.)
@dataclass
class State:
 n:int=0
 sums:dict=field(default_factory=lambda:dict.fromkeys(KEYS,0.))
 def snap(self,p):
  w=p+2/(1+self.sums['opponent_n']/max(self.n,1))
  return {k:(self.sums[k]+w*DEFAULT[k])/(self.n+w) for k in KEYS}|dict(n=self.n)
def features(a,b,v,d,p):
 a,b=a.snap(p),b.snap(p);off=a['off']-b['off'];de=b['def']-a['def'];net=off+de;pace=(a['pace']+b['pace'])/2;n=min(a['n'],b['n']);turn=(a['deftov']+b['tov'])-(b['deftov']+a['tov']);glass=(a['orb']+b['deforb'])-(b['orb']+a['deforb']);shot=(a['efg']+b['defefg'])-(b['efg']+a['defefg']);foul=(a['ftr']+b['defftr'])-(b['ftr']+a['defftr']);z=net*pace/100
 return dict(venue=v,off=off,de=de,net=net,shot=shot,turn=turn,glass=glass,foul=foul,sos=a['sos']-b['sos'],pace=pace,n=n,day=d,net_pace=z,net_tail=z*abs(z)/20,net_early=z/(1+n/5),venue_early=v/(1+n/5),turn_glass=(turn+glass)*abs(net),shot_net=shot*abs(net))
def apply(g,a,b,ap,bp,p):
 poss=float(g.poss);pace=poss*40/(40+5*g.num_ot)
 def factors(x,y):
  fga=getattr(g,x+'_fga');fta=getattr(g,x+'_fta');ora=getattr(g,x+'_or');tov=getattr(g,x+'_to')
  return (getattr(g,x+'_fgm')+.5*getattr(g,x+'_fgm3'))/max(fga,1),tov/max(fga+.475*fta-ora+tov,1),ora/max(ora+getattr(g,y+'_dr'),1),fta/max(fga,1)
 for s,op,x,y,sc,opp_sc in [(a,bp,'a','b',g.score_a,g.score_b),(b,ap,'b','a',g.score_b,g.score_a)]:
  e,t,r,f=factors(x,y);oe,ot,orr,of=factors(y,x);coeff=.85*op['n']/(op['n']+p)
  obs=dict(off=sc/poss*100-coeff*(op['def']-101),def_=opp_sc/poss*100-coeff*(op['off']-101),pace=pace,efg=e,tov=t,orb=r,ftr=f,deftov=ot,deforb=orr,defefg=oe,defftr=of,sos=op['off']-op['def'],opponent_n=op['n']);obs['def']=obs.pop('def_')
  for k,val in obs.items():s.sums[k]+=val
  s.n+=1
def build(games,p,emit=True):
 rows=[];states={}
 for season,sg in games.groupby('season',sort=True):
  states={}
  for d,group in sg.groupby('day_num',sort=True):
   pending=[]
   for g in group.itertuples(index=False):
    a=states.setdefault(g.team_a,State());b=states.setdefault(g.team_b,State())
    if emit:rows.append(dict(season=season,day_num=d,team_a=g.team_a,team_b=g.team_b,margin=g.margin,**features(a,b,g.venue_a,d,p)))
    pending.append((g,a,b,a.snap(p),b.snap(p)))
   for g,a,b,ap,bp in pending:apply(g,a,b,ap,bp,p)
 return pd.DataFrame(rows) if emit else states
