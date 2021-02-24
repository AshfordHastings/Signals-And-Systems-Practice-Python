# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:55:58 2021

@author: William
"""

import numpy as np
import matplotlib as plt
import math
from IPython.display import Audio

class Signal:
    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        n = round(duration * framerate)
        ts = start + np.arange(n) / framerate
        ys = self.evaluate(ts)
        return Wave(ys, ts, framerate=framerate)

    def plot(self, duration=None, period_length=3, **options):
        if duration is None:
            duration = self.period * period_length
        wave = self.make_wave(duration)
        wave.plot()

class Sinusoid(Signal):
    """Represents sinusoidal signal"""
    
    def __init__(self, freq, amp, offset, func):
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func
    
    ##Get function
    @property
    def period(self):
        return 1.0 / self.freq
    
    ## Evaluate - producing the signal from the properties
    def evaluate(self, ts):
        """ts is a float array of times that would come from the Wave class"""
        "phase = 2*pi*f*t + offset"
        ts = np.asarray(ts)
        phases = 2*math.pi * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys
    
def CosSignal(freq, amp=1.0, offset=0):
    return Sinusoid(freq, amp, offset, func=np.cos)

def SinSignal(freq, amp=1.0, offset=0):
    return Sinusoid(freq, amp, offset, func=np.sin)

class Wave:
    """Class is meant to take in a Signal and return a certain wave interval"""
    def __init__(self, ys, ts=None, framerate=None):
        self.ys = np.asanyarray(ys)
        "Initializes framerate default value"
        self.framerate = framerate if not None else 11025
        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)
            
    def plot(self, **options):
        """plots the real value of the Wave"""
        """**args passes a dictionary of arguments that maps to values - can be inserted into plot()"""
        plt.plot(self.ts, np.real(self.ys), **options)
    
    def make_audio(self):
        return Audio(data=self.ys.real, rate=self.framerate)
    

