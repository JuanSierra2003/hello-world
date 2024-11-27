import numpy as np
from scipy.integrate import quad

def gaussian(x:float, mu:float, sigma:float):
    return np.exp(-.5*((x-mu)/sigma)**2)

def lorentzian(x:float, x0:float, tau:float):
    return (1+((x- x0)/tau)**2)**-2

curves = {'gaussian':gaussian, 'lorentzian':lorentzian}
fmwh_const = {'gaussian':2.355, 'lorentzian':1.287}

class pulse():

    def __init__(self, spectrum, w0:float, spec_width:float):
        self.spectrum = spectrum
        self.w0 = w0
        self.spec_width = spec_width

    def set_linear_chirp(self, alphap:float):
        aux_func = self.spectrum
        self.spectrum = lambda w: aux_func(w)*np.exp(-.5j*alphap*(w-self.w0)**2)

    def set_noch(self, curve_type:str, fwhm:float):
        if fwhm > 0:
            aux_func = self.spectrum
            self.spectrum = lambda w: aux_func(w)*(1 - curves[curve_type](w, self.w0, fwhm/fmwh_const[curve_type]))

    def time_dependant_field(self, t:float):
        func = lambda w: self.spectrum(w)*np.exp(-1j*w*t)
        return quad(func, self.w0-2*self.spec_width, self.w0+2*self.spec_width, complex_func=True, epsabs=1.49e-09)[0]
    
    def time_dependant_field_fft(self, n_points: int = 2048, eps: float = 1e-10):
        N = 1e2
        """
        Reconstruct the time-domain field using FFT.
        
        Args:
            t_values (np.ndarray): Array of time points where the field should be computed.
            n_points (int): Number of points in the FFT grid.
            
        Returns:
            np.ndarray: Time-domain field values at the specified time points.
        """
        # Define the frequency grid
        dw =  6 * self.spec_width / n_points # Frequency resolution
        w = np.linspace(self.w0 - 3*self.spec_width, self.w0 + 3*self.spec_width, n_points)
        
        # Sample the spectrum
        spectrum_vals = np.array([self.spectrum(wi) for wi in w])
        
        # Perform the inverse FFT to compute the time-domain field
        time = np.fft.fftfreq(n_points*100, d=dw) * 2 * np.pi # Convert to time
        field_fft = np.fft.ifft(spectrum_vals, n=n_points*100)
        
        # Shift the result to align the time grid with FFT convention
        time = np.fft.fftshift(time)
        field_fft = np.fft.fftshift(field_fft)
        
        # Interpolate the field at the requested time points

        pick = np.max(np.abs(field_fft))
        mask = (np.abs(field_fft)/pick > eps)
        
        return time[mask], field_fft[mask]
    
class gaussian_pulse(pulse):

    def __init__(self, w0: float, fwhm: float, area: float):
        spectrum = lambda w: gaussian(w, w0, fwhm / 2.355) * area * np.sqrt(2 * np.pi)
        super().__init__(spectrum, w0, fwhm)
