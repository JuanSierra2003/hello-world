import numpy as np

#Inverse Fourier transform by its definition
def inverse_fourier(w_range: tuple, signal_fn , num_intervals: int, time: float):
    w = np.linspace(*w_range, num_intervals)
    dw = w[1] - w[0]
    return np.sum(signal_fn(w) * np.exp(-1j * w * time)) * dw

#Some pulse types
def gaussian(x:float, mu:float, sigma:float):
    return np.exp(-.5*((x-mu)/sigma)**2)

def lorentzian(x:float, x0:float, tau:float):
    return (1+((x- x0)/tau)**2)**-1

def square(x: float, x0: float, delta: float, smoothness: float = .5):
    left_edge = 1 / (1 + np.exp(-(x - (x0 - delta))/smoothness))
    right_edge = 1 / (1 + np.exp((x - (x0 + delta))/smoothness))
    return left_edge*right_edge

def sech(x:float, x0:float, tau):
    return 1/np.cosh((x-x0)/tau)

#Dictionarys for selecting pulse type
curves = {'gaussian':gaussian, 'lorentzian':lorentzian, 'square':square, 'sech':sech}

fmwh_const = {'gaussian':np.sqrt(np.log(16)), 'lorentzian':2*np.sqrt(np.sqrt(2)-1), 
              'square':1, 'sech':2*np.log(np.sqrt(2)-1)}

#Constructor class of the pulses
class pulse():

    #Initialization
    def __init__(self, spectrum, w0:float, spec_width:float):
        self.spectrum = spectrum                                #Spectrum without chirp nor notch
        self.notch = lambda w: 1                                #Nocht function initialized as 1
        self.chirp = lambda w: 1                                #Chirp function initialized as 1
        self.w0 = w0                                            #Central frequency
        self.spec_width = spec_width                            #spec_width

    #Setting a certain linear chirp 
    def set_linear_chirp(self, alphap:float):
        if alphap != None:
            self.chirp = lambda w: np.exp(.5j*alphap*(w-self.w0)**2)
        else:
            self.chirp = lambda w: 1

    #Setting a certain type of notch with width fmwh
    def set_notch(self, curve_type:str, fwhm:float):
        if fwhm > 0 and curve_type != None:
            self.notch = lambda w: 1 - curves[curve_type](w, self.w0, fwhm/fmwh_const[curve_type])
        elif curve_type == None:
            self.notch = lambda w: 1
        else: 
            pass

    #Returns the field as a function of time
    def time_dependant_field(self, num_intervals: int = int(1e3)):
        w_range = (self.w0 - 5*self.spec_width, self.w0 + 5*self.spec_width)
        signal_fn = lambda w: self.spectrum(w)*self.chirp(w)*self.notch(w)

        return np.vectorize(lambda time: inverse_fourier(w_range, signal_fn, num_intervals, time))

#Constructor of a gaussian pulse
class gaussian_pulse(pulse):

    #Initialization
    def __init__(self, w0: float, fwhm: float, area: float):
        spectrum = lambda w: gaussian(w, w0, fwhm / 2.355) / (2 * np.pi) * area
        super().__init__(spectrum, w0, fwhm)
