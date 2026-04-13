"""
Code to compute the cooling coefficient for blackbody radiation 
around a binary with given mass and semi-major axis
"""

from logging import getLogger
from typing import NamedTuple
from math import sqrt
import numpy as np

pi = 3.14159265359
cgs = dict(
		G = 6.6725985e-8,
		c = 2.99792458e10,
		kb = 1.38065812e-16,
		sigmab = 5.6705119e-5,
		mp = 1.6726e-24,
		kappa = 0.4,            # electron scattering
		pc = 3.085678e18,
		msun = 1.989e33,
		h = 6.62607015e-27,
		blackbodyconst1 = 1.4745e-47,
		blackbodyconst2 = 4.79921e-11,
		c2h3 = 2.61463e-58,
		h_over_kb = 4.79921e-11,
		year=31556952,
		ev = 1.60218e-12,
		q_e = 4.80320425e-10,
	)

logger = getLogger(__name__)

class ShakuraSunyaevDisk(NamedTuple):
	"""The central mass, length scale (e.g. binary separation), alpha, and mach number (at r=a)
	   for a Shakura-Sunyaev alpha-disk (Frank, King, & Raine 2002) 
		- for numerical reasons the disk Mach number at r=a is supplied (as oppoed to the
		  accretion rate), and mdot as a fraction of the eddington rate is determined 
		  accordingly.
		- the target_accretion_rate is supplied to package the accretion rate remapping together
		  with the disk structure as a single container and for clarity in checkpoints.
	"""

	central_mass_msun     : float
	length_scale_pc       : float
	mach_number_a         : float
	alpha                 : float
	gamma                 : float
	target_accretion_rate : float

	# -------------------------------------------------------------------------
	@property
	def _mass(self) -> float:
		return self.central_mass_msun * cgs['msun']

	@property
	def _length(self) -> float:
		return self.length_scale_pc * cgs['pc']
	
	@property
	def _time(self) -> float:
		return sqrt(self._length**3 / self._GM)  # Omega_b^-1

	@property
	def _GM(self) -> float:
		return self._mass * cgs['G']	
	
	@property
	def _rschwz(self):
		return 2 * self._GM / cgs['c']**2

	@property
	def _accretion_efficiency(self):
		"""
		For a compact object; eta = 0.1, 0.15
		"""
		return 0.1
	
	@property
	def _eddington_rate(self) -> float:
		"""
		Here: kappa = thompson / mass proton
		"""
		return 4 * pi * self._GM / cgs['kappa'] / cgs['c'] / self._accretion_efficiency


	@property
	def _eddington_fraction(self) -> float:
		"""
		Calculate the eddington fraction of a reference Shakura Sunyaev disk at r=a. 

		   f_edd = √(32𝛑^2/3) * (mp^4 / kb^4 * sigmab / kappa * alpha)^(1/2) * (GM)^(7/4) * Mach(r)^-5 * r^(-1/4) / Mdot_edd

		This fraction of the eddington rate is returned
		"""
		f0 = 10.2604 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**0.5 * self.gamma**(-2.)
		return f0 * self.alpha**0.5 * self._GM**(7./4.) * self._length**(-1./4.) * self.mach_number_a**-5 / self._eddington_rate
	
	@property
	def _accretion_rate(self) -> float:
		"""
		The actual accretion rate, parameterised by the Eddington rate
		"""
		return self._eddington_fraction * self._eddington_rate
	
	@property
	def Mdrop(self):
		return self.target_accretion_rate / self._eddington_fraction

	@property
	def _surface_density(self) -> float:
		"""
		Disk surface density from Shakura Sunyaev in cgs

		   Sigma = (32 / 3^6 pi^3)^(1/5) * (mp^4 / kb^4 * sigmab / kappa)^(1/5) 
		   			* alpha^(-4/5) * (GM)^(1/5) * Mdot^(3/5) * r^(-3/5)
		"""
		s0 = 0.269274 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.) * self.gamma**(-4./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)

	@property
	def _surface_pressure(self) -> float:
		"""Disk pressure scaling in cgs

		   P = (1 / 3 / pi) * alpha^-1 * Mdot * (GM)^0.5 * r^(-3/2)
		"""
		return 0.106103 / self.gamma / self.alpha * self._accretion_rate * sqrt(self._GM) * self._length**(-3./2.)

	@property
	def _midplane_temperature(self) -> float:
		"""Disk midplane temperature scaling in cgs (for completeness)

		   T = (3 / 32 / pi^2)^(1/5) * (mp * kappa / kb / sigmab)^(1/5) 
		        * alpha^(-1/5) * (GM)^(3/10) * Mdot^(2/5) * r^(-9/10)
		"""
		t0 = 0.394035 * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.) * self.gamma**(-1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	# -------------------------------------------------------------------------
	@property
	def surface_density_coefficient(self) -> float:
		return self._surface_density / (self._mass / self._length**2)

	@property
	def surface_pressure_coefficient(self) -> float:
		return self._surface_pressure / (self._mass / self._time**2)	

	def surface_density_profile(self, r:float) -> float:
		return self.surface_density_coefficient * r**(-3./5.)

	def surface_pressure_profile(self, r:float) -> float:
		return self.surface_pressure_coefficient * r**(-3./2.)

	def mach_profile(self, r:float) -> float:
		cs    = (self.gamma * (self.surface_pressure_profile(r) / self.surface_density_profile(r)))**0.5
		Omega = r**(-3./2.)
		Hs    = cs /Omega
		return r / Hs

	def Effective_optical_depth(self, r:float) -> float:
		H     = r * self._length / self.mach_profile(r)  # in cgs 
		Sigma = self.surface_density_profile(r)  * (self._mass / self._length**2)       
		Press = self.surface_pressure_profile(r) * (self._mass / self._time**2)	
		T     = (Press / Sigma) * (cgs['mp'] / cgs['kb'])
		rho   = Sigma / H

		# page 163 Radiative Processes in Astrophysics, Rybicki & Lightman (1986)
		ne                 = rho / cgs['mp']
		ni                 = ne
		Z                  = 1.0
		gaunt_r            = 1.0
		alpha_ff           = 1.7e-25 * T**(-7/2) * Z**2 * ne * ni * gaunt_r  # cgs units
		tau_ff             = alpha_ff * H
		tau_es             = cgs['kappa'] * Sigma

		return np.sqrt(tau_ff * (tau_ff + tau_es))



	def optical_depth(self, r:float) -> float:
		return cgs['kappa'] * self.surface_density_profile(r) * (self._mass / self._length**2)

	# -------------------------------------------------------------------------
	@property
	def cooling_coefficient(self) -> float:
		"""
		Assumes avg fluid particle mass is the proton mass
	
				P / Sigma = eps * (gamma - 1)
	
				deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4
	
				eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 
	
				cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb) * (gamma - 1)
		"""
		mp_code = cgs['mp'] /  self._mass
		kb_code = cgs['kb'] / (self._mass * self._length**2 / self._time**2)
		kappa_code  = cgs['kappa'] / (self._length**2 / self._mass)
		sigmab_code = cgs['sigmab'] / (self._mass / self._time**3)	
		qdot_coeff = 8. / 3. * sigmab_code / kappa_code * (mp_code / kb_code)**4 * (self.gamma - 1.)**4
		return qdot_coeff

	# Only for temporary testing
	# =============================================================================
	def surface_density_goodman(self):
		coeff = 2**(4./5.) / 3. / pi**(3./5.)
		s0 = coeff * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)
		
	def midplane_temperature_goodman(self):
		coeff = (1. / 16. / pi**2)**(1./5.)
		t0 = coeff * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	def surface_pressure_goodman(self):
		return cgs['kb'] / cgs['mp'] * self.midplane_temperature_goodman() * self.surface_density_goodman()
	
	def Mdot(self, r):
		cs    = (self.gamma * (self.surface_pressure_profile(r) / self.surface_density_profile(r)))**0.5
		Omega = r**(-3./2.)
		Hs    = cs / Omega
		nu    = self.alpha * cs * Hs
		return 3 * pi * self.surface_density_profile(r) * nu
	
	@property
	def Mdot_inf(self):
		return self.Mdot(1.0)
		
	# =============================================================================
	# ========================== Code unit conversions ============================
	# =============================================================================

	@property
	def kb_code(self):
		return cgs['kb'] / (self._mass * self._length**2 / self._time**2)
	
	@property
	def sigmab_code(self):
		return cgs['sigmab'] / (self._mass / self._time**3)	

	@property
	def mp_code(self):
		return cgs['mp'] / (self._mass)

	@property
	def kappa_code(self):
		return cgs['kappa'] / (self._length**2 / self._mass)

	@property
	def ff_absorption_code(self):
		# page 163 Radiative Processes in Astrophysics, Rybicki & Lightman (1986)
		# (1) convert number density to mass density by dividing prefactor by cgs['mp']^2 
		# (2) convert from cgs to code units by multiplying by (SS73._mass^2 / SS73._length^5)
		prefactor          = 1.7e-25 / cgs['mp'] / cgs['mp']
		prefactor_code     = prefactor * (self._mass**2 / self._length**5)
		return prefactor_code

	@property   
	def Length_Scale_CGS(self): # physical units (not code units)
		return self.length_scale_pc * cgs['pc']


def EffectiveTemperature(optical_depth, T):
	"""
	When we obtain a solution in the midplane, we need to note that the disk is cooling 
	throught the surface by sigma * Teff ^4. This surface, (or effective) temperature is is used
	to calculate the energy being radiated away from the surface. Consistenly solving the vertical 
	energy flux would require more sophisticated 3D modelling.
	""" 
	return T * (4./3./optical_depth)**0.25

# Precompute the fixed values for the infrared and optical bands
nu_infared_low  = cgs['c'] / (0.1)     # 1 mm = 0.1 cm
nu_infared_high = cgs['c'] / (7e-5)    # 700 nm = 7e-5 cm

nu_optical_low  = cgs['c'] / (7e-5)    # 700 mm = 7e-5 cm
nu_optical_high = cgs['c'] / (4e-5)    # 400 nm = 4e-5 cm

nu_UV_low       = cgs['c'] / (4e-5)
nu_UV_high      = cgs['c'] / (1e-6)

nu_Xray_low     = cgs['c'] / (1e-6)
nu_Xray_high    = cgs['c'] / (1e-9)

E_Xray_low   , E_Xray_high    = cgs['h'] * nu_Xray_low   , cgs['h'] * nu_Xray_high
E_UV_low     , E_UV_high      = cgs['h'] * nu_UV_low     , cgs['h'] * nu_UV_high
E_optical_low, E_optical_high = cgs['h'] * nu_optical_low, cgs['h'] * nu_optical_high
E_infared_low, E_infared_high = cgs['h'] * nu_infared_low, cgs['h'] * nu_infared_high

def Energy_to_wavelength(E):
	return (cgs['h'] * cgs['c']) / E 

def Wavelength_to_energy(L):
	return (cgs['h'] * cgs['c']) / L 

def BandEmission(temperature, nu_low, nu_high):
    x_low  = cgs['h_over_kb'] * nu_low  / temperature
    x_high = cgs['h_over_kb'] * nu_high / temperature
    x_grid = np.logspace(np.log10(x_low), np.log10(x_high), 400)
    x_grid = np.clip(x_grid, None, 700.0)

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        integrand = np.where(x_grid > 700.0, 0.0, x_grid**3 / np.expm1(x_grid))

    integral  = np.trapezoid(integrand, x_grid, axis=0)
    prefactor = (2 * (cgs['kb'] * temperature)**4) / cgs['c2h3']
    return np.pi * prefactor * integral

def InfaredEmission(temperature):  
	return BandEmission(temperature, nu_infared_low, nu_infared_high)

def OpticalEmission(temperature):  
	return BandEmission(temperature, nu_optical_low, nu_optical_high)

def UVEmission(temperature):       
	return BandEmission(temperature, nu_UV_low, nu_UV_high)

def XrayEmission(temperature):     
	return BandEmission(temperature, nu_Xray_low, nu_Xray_high)

def PlanckSpectrum(nu, T):
    B_nu = (2 * cgs['h'] * nu**3 / cgs['c']**2) / np.expm1(cgs['h'] * nu / (cgs['kb'] * T))
    return np.pi * B_nu

def TransitionRadii(m, alpha, mdot, a): # m in units of Msun, mdot in units of mdot_Edd and a in units of Rgrav
    # (1) Radiation pressure -> Gas pressure transition
    
    C           = 150 * (alpha * m)**(2/21) * mdot**(16/21)
    x           = C                                          # initial guess
    f_x         = x / (1-x**(-0.5))**(16/21) - C
    iteration   = 0
    vals        = []
    while np.abs(f_x)>1e-12 and iteration<10:
        derivative   = ((1 - x**(-0.5)) - 8 / 21 / x**(0.5)) / (1-x**(-0.5))**(37/21)
        x1           = x  - (f_x / derivative)
        f_x1         = x1 / (1-x1**(-0.5))**(16/21) - C
        x            = np.copy(x1)
        f_x          = np.copy(f_x1)
        iteration   += 1
        vals.append(f_x)

    print('===== Radiation -> Gas Pressure =====')
    print('Transition in Rgrav.........',x * 3)
    print('Ratio to semimajor axis.....', x * 3 / a)
    print('Approximate Error...........', f_x)

    # (2) Electron scattering -> Free free absorption transition
    C           = 6.3e3 * mdot**(2/3)
    x           = C                                       # initial guess
    f_x         = x / (1-x**(-0.5))**(2/3) - C
    iteration   = 0
    vals        = []
    while np.abs(f_x)>1e-12 and iteration<10:
        derivative   = ((1 - x**(-0.5)) - 1 / 3 / x**(0.5)) / (1-x**(-0.5))**(5/3) 
        x1           = x  - (f_x / derivative)
        f_x1         = x1 / (1-x1**(-0.5))**(2/3) - C
        x            = np.copy(x1)
        f_x          = np.copy(f_x1)
        iteration   += 1
        vals.append(f_x)

    print('===== es -> ff opacity transition =====')
    print('Transition in Rgrav.........',x * 3)
    print('Ratio to semimajor axis.....', x * 3 / a)
    print('Approximate Error...........', f_x)


if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	r  = np.linspace(0.5, 10., 250)
	ss = ShakuraSunyaevDisk(
        	central_mass_msun = 8e6, 
        	length_scale_pc   = 9.7e-4,
        	mach_number_a     = 7,
        	alpha             = 0.1,
			gamma             = 5./3.,
			target_accretion_rate=10.0,
        )
	print("fedd     : ", ss._eddington_fraction)
	print("Sigma    : ", ss.surface_density_profile(1.0))
	print("Pressure : ", ss.surface_pressure_profile(1.0))
	mp_code = cgs['mp'] /  ss._mass
	kb_code = cgs['kb'] / (ss._mass * ss._length**2 / ss._time**2)


	fcavity = 0.0001 + 0.9999 * np.exp(-((1.0 / r) ** 30))
	fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, figsize=[7,8])
	ax1.plot(r, ss.surface_density_profile(r) * fcavity, c='C0')
	ax1.plot(r, ss.surface_density_goodman() / (ss._mass / ss._length**2) * r**(-3./5.) * fcavity, c='C1', ls='--')
	ax1.plot(r, 0.057 * r**(-3./5.) * fcavity, c='C3')

	ax2.plot(r, ss.surface_pressure_profile(r) * fcavity, c='C0')
	ax2.plot(r, ss.surface_pressure_goodman() / (ss._mass / ss._time**2) * r**(-3./2.) * fcavity, c='C1', ls='--')
	ax2.plot(r, pi * 6.7e-5 * r**(-3./2.) * fcavity, c='C3')

	ax3.plot(r, ss.surface_pressure_profile(r) / ss.surface_density_profile(r), c='C0')
	ax3.plot(r, 6.7e-5 * r**(-3./2.) / (0.057 * r**(-3./5.)), c='C3')

	ax4.plot(r, ss.mach_profile(r), c='silver')
	ax4.set_ylim([0, ss.mach_profile(0.1)])

	ax1.set_ylabel(r'$\Sigma$')
	ax2.set_ylabel(r'$P$')
	ax3.set_ylabel(r'$c_s^2$')
	ax4.set_ylabel(r'$\mathcal{M}$')
	ax3.set_xlabel(r'$r$')
	ax4.set_xlabel(r'$r$')

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.1)
	plt.show()

	plt.figure()
	plt.plot(r, 2 * np.pi * ss.Mdot(r), c='C0')
	plt.ylim([-0.5,0.5])
	plt.savefig('mdot_profile.png')

