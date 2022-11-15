import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy.optimize import curve_fit
from lmfit import Model
from symfit import parameters, variables, sin, cos, Fit

def nufit_fourier(x, y):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''

    x = np.array(x)
    y = np.array(y)

    freq = np.fft.fftfreq(len(x), (x[1]-x[0]))   # assume uniform spacing
    Fy = abs(np.fft.fft(y))

    guess_freq = abs(freq[np.argmax(Fy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c

    popt, pcov = curve_fit(sinfunc, x, y, p0=guess, maxfev = 10000)
    A, w, p, c = popt
    #f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c

    #return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    return fitfunc(x)

def lmfit_fourier(x,y):

    x = np.array(x)
    y = np.array(y)

    freq = np.fft.fftfreq(len(x), (x[1]-x[0]))   # assume uniform spacing
    Fy = abs(np.fft.fft(y))

    guess_freq = abs(freq[np.argmax(Fy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    #guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(x, A, w, p, c):  return A * np.sin(w*x + p) + c

    gmodel = Model(sinfunc)
    params = gmodel.make_params(A = guess_amp, w = 2.*np.pi*guess_freq, p = 0., c = guess_offset)
    sinus_fit = gmodel.fit(y, params, x=x)

    return sinus_fit.best_fit

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def symfit_fourier(freq,signal):

    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, f=w, n=3)}

    # Define a Fit object for this model and data
    fit = Fit(model_dict, freq, signal)
    fit_result = fit.execute()

    return fit.model(x=freq, **fit_result.params).y

def get_peaks(freq, signal, method = 'numpy'):

    if method == 'numpy':

        try: 

            fit = nufit_fourier(freq,signal)

            min_peaks = argrelmin(fit)[0]
            min_x = freq[min_peaks]
            min_y = fit[min_peaks]

            max_peaks = argrelmax(fit)[0]
            max_x = freq[max_peaks]
            max_y = fit[max_peaks]

        except Exception as e:

            print(e)
            min_x, min_y, max_x, max_y = None, None, None, None

    elif method == 'lm':

        try: 

            fit = lmfit_fourier(freq,signal)

            min_peaks = argrelmin(fit)[0]
            min_x = freq[min_peaks]
            min_y = fit[min_peaks]

            max_peaks = argrelmax(fit)[0]
            max_x = freq[max_peaks]
            max_y = fit[max_peaks]

        except Exception as e:

            print(e)
            min_x, min_y, max_x, max_y = None, None, None, None

    elif method == 'sym':
        
        try:

            fit = symfit_fourier(freq,signal)

            min_peaks = argrelmin(fit)[0]
            min_x = freq[min_peaks]
            min_y = fit[min_peaks]

            max_peaks = argrelmax(fit)[0]
            max_x = freq[max_peaks]
            max_y = fit[max_peaks]

        except Exception as e:

            print(e)
            min_x, min_y, max_x, max_y = None, None, None, None

    return (min_x, min_y), (max_x, max_y)