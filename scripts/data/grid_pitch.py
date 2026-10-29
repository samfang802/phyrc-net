#!/usr/bin/env python3
"""Measure grid-line pitch (px) from early undamaged frames: autocorrelation of dark-line projections."""
import cv2, numpy as np, glob, sys, os
D="Data/DamagePattern/images/"
def lines(g):
    # dark thin lines: black-hat
    bh=cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(21,21)))
    return bh.astype(np.float32)
def pitch(sig, lo=20, hi=300):
    s=sig-sig.mean(); n=len(s)
    ac=np.correlate(s,s,'full')[n-1:]; ac/=ac[0]
    # first strong peak after the first minimum
    k=lo+np.argmax(ac[lo:hi]); 
    # refine parabolic
    if 1<k<n-1:
        a,b,c=ac[k-1],ac[k],ac[k+1]; k=k+0.5*(a-c)/(a-2*b+c)
    return k, ac
if __name__=="__main__":
    for f in sys.argv[1:]:
        g=cv2.imread(f,0); bh=lines(g)
        px,_=pitch(bh.sum(0)); py,_=pitch(bh.sum(1))
        print(os.path.basename(f), g.shape, round(px,1), round(py,1))
