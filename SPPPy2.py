# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:56:28 2021.

@author: Tigrus

Library for calculus in multilayers scheme with gradient layer v 2.0 release
ver 11.08.2021
"""


from .SuPPPort import *
import copy

class ExperimentSPR:
    """Experiment class for numeric calculus."""

    gradient_resolution = 100  # resolution for gradient layer calculation
    wavelength = 0
    fix_granient_n_minimum = False
    def __init__(self, polarisation='p'):
        """Init empty."""
        self.layers = dict()
        self.k0 = 2.0 * np.pi / 1e-6
        self.polarisation = polarisation

        # for system conserving
        self.wl_asylum = dict()
        self.k0_asylum = 1e-6

    def __setattr__(self, name, val):
        """Sync wavelength and k0."""
        if name == "k0":
            self.__dict__["k0"] = val
            self.__dict__["wavelength"] = 2 * np.pi / val
        elif name == "wavelength":
            self.__dict__["wavelength"] = val
            self.__dict__["k0"] = 2 * np.pi / val
        else: self.__dict__[name] = val

    def __getattr__(self, attrname):
        """Getter for n and d."""
        if attrname == "n":
            val = []
            for L in range(0, len(self.layers)):
                if isinstance(self.layers[L].n, MaterialDispersion) or\
                    isinstance(self.layers[L].n, LorentzDrude):
                    # Parametric metal
                    val.append(self.layers[L].n.CRI(self.wavelength))
                elif isinstance(self.layers[L].n, FunctionType):
                    # Gradient layer fubction
                    val.append(self.layers[L].n(0))
                elif isinstance(self.layers[L].n, Anisotropic):
                    # Anisotropic layer fubction
                    val.append(self.layers[L].n.n0)
                else:
                    # Homogenious
                    val.append(self.layers[L].n)
            return val
        if attrname == "d":
            val = [0]
            for L in range(1, len(self.layers)-1):
                val.append(self.layers[L].thickness)
            val.append(0)
            return val

    def save_scheme(self):
        """Conserving scheme parametrs."""
        self.layers_asylum = copy.deepcopy(self.layers)
        self.k0_asylum = self.k0

    def load_scheme(self):
        """Rescuing scheme parametrs."""
        self.layers = copy.deepcopy(self.layers_asylum)
        self.k0 = self.k0_asylum

    # -----------------------------------------------------------------------
    # --------------- Work with layers --------------------------------------
    # -----------------------------------------------------------------------

    def add(self, new_layer):
        """Add one layer.

        Parameters
        ----------
        permittivity : complex, Metall_CRI or lambda
            permittivity for layer.
        thickness : float
            layer thickness.
        """
        self.layers[len(self.layers)] = new_layer

    def delete(self, num):
        """Delete one layer.

        Parameters
        ----------
        num : int
            layer number.
        """
        if num < 0 or num > len(self.layers)-1:
            print("Deleting layer out of bounds!")
            return
        if num == len(self.layers) - 1:
            self.layers.pop(num)
        else:
            for i in range(num, len(self.layers)-1):
                self.layers[i] = self.layers.pop(i+1)

    def insert(self, num, new_layer):
        """Insert layer layer

        Parameters
        ----------
        num : int
            layer number to insert.
        new_layer : [array]
            [permittivity, thickness]
        """
        if num < 0 or num > len(self.layers)-1:
            print("Inserting layer out of bounds! Layer add in the end of the list")
            self.add(new_layer)
        else:
            for i in range(0, len(self.layers) - num + 1):
                self.layers[len(self.layers) - i] = self.layers[len(self.layers) - i - 1]
            self.layers[num] = new_layer

    # -----------------------------------------------------------------------
    # --------------- Profiles calculations ---------------------------------
    # -----------------------------------------------------------------------

    def R(self, angles=None, wavelenghts=None, angle=None, is_complex=False,
          spectral_width=0, spectral_resolution=20):
        """Representation for every R.

        Parameters
        ----------
        angles : arary, optional
            angles range. The default is None.
        wavelenghts : arary, optional
            wavelenghts range. The default is None.
        angle : float, optional
            angle for r(lambda). The default is None.
        is_complex : boolean, optional
            return real or complex. The default is false.

        Returns
        -------
        arary
            array of R.

        """

        # Ordinary R
        if spectral_width == 0:
            if angles is not None:
                if is_complex:
                    return self.R_theta_cmpl(angles)
                else:
                    return self.R_theta_re(angles)
            if wavelenghts is not None:
                if angle is None:
                    print("Angle not defined!")
                    return None
                if is_complex:
                    return self.R_lambda_cmpl(angle, wavelenghts)
                else:
                    return self.R_lambda_re(angle, wavelenghts)   
            print("Parametrs do not defined!")
            return
        
        if angles is not None:
            # R with spectral width
            spectral_function = lambda x: spectral_width / (2 * np.pi * 
                           ((self.wavelength - x)**2 + spectral_width**2 / 4))
            wavelenghts_list = np.linspace(self.wavelength - spectral_width * 2,
                                     self. wavelength + spectral_width * 2,
                                      spectral_resolution * 2 - 1)
            intensities = spectral_function(wavelenghts_list)
            intensities_sum = sum(intensities)
            
            R_collect = np.zeros(len(angles))
            self.save_scheme()
            for i in range(len(wavelenghts_list)):
                self.wavelength = wavelenghts_list[i]
                R_collect = R_collect + np.array(self.R_theta_re(angles)) \
                    *(intensities[i]/intensities_sum)
            self.load_scheme()
            return(R_collect)

    def R_theta_re(self, degree_range):
        """Parameters.

        degree_range : range(start, end, seps)
            range of function definition in degree .

        Returns
        -------
        Rr : array[float]
            reflection array in range.
        """
        Rr = [np.abs(self.R_deg(theta))**2 for theta in degree_range]
        return Rr

    def R_theta_cmpl(self, degree_range):
        """Parameters.

        degree_range : range(start, end, seps)
            range of function definition in degree .

        Returns
        -------
        Rr : array[complex]
            reflection array in range.
        """
        Rr = [self.R_deg(theta) for theta in degree_range]
        return Rr

    def R_lambda_re(self, angle_grad, lambda_range):
        """Parameters not work yet.

        angle_grad : float
            angle of calculus in [0 - 90]
        lambda_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[float]
            reflection array in range.
        """
        Rr = []
        for i in lambda_range:
            self.wavelength = i
            Rr.append(np.abs(self.R_deg(angle_grad))**2)
        return Rr

    def R_lambda_cmpl(self, angle_grad, lambda_range):
        """Parameters not work yet.

        angle_grad : float
            angle of calculus in [0 - 90]
        lambda_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[complex]
            reflection array in range.
        """
        Rr = []
        for i in lambda_range:
            self.wavelength = i
            Rr.append(self.R_deg(angle_grad))
        return Rr

    def R_deg(self, theta):
        """Parameters.

        theta : int
            angle to calculate on.

        Returns.
        -------
        R : float
            complex reflection in selected angle.
        """
        theta = np.pi * theta / 180
        n = self.n
        d = self.d
        kx0_sqrt = self.k0 * n[0] * np.sin(theta)
        kx0 = np.power(kx0_sqrt, 2)

        k_z = [SM.sqrt(np.power(self.k0*n[i], 2) - kx0) for i in range(1, len(n))]

        # k_z[grad] valid only for input!
        k_z.insert(0, np.sqrt(np.power(self.k0*n[0], 2) - kx0))
 
        # find r and t for normal layers
        if self.polarisation == 'p':
            rp = [(k_z[i]*n[i+1]**2 - k_z[i+1]*n[i]**2) /
                 (k_z[i]*n[i+1]**2 + k_z[i+1]*n[i]**2)
                 for i in range(0, len(n)-1)]

            tp = [2*(k_z[i]*n[i]**2) /
                 (k_z[i]*n[i+1]**2 + k_z[i+1]*n[i]**2)
                 for i in range(0, len(n)-1)]

        if self.polarisation == 's':
            rs = [(k_z[i] - k_z[i+1]) /
                 (k_z[i] + k_z[i+1])
                 for i in range(0, len(n)-1)]

            ts = [2*(k_z[i]) /
                 (k_z[i] + k_z[i+1])
                 for i in range(0, len(n)-1)]

        # All layers for p
        if self.polarisation == 'p':
            # reflectivities for anisotropic layers for this angles for 'p'
            # 's' dont feels extraordinary n
            for i in range(1, len(n)-1):
                if isinstance(self.layers[i].n, Anisotropic):
                    # before layer
                    if not isinstance(self.layers[i - 1].n, Anisotropic):
                        # if previous is anisotropic - r  is modified in prev step
                        rp[i-1] = self.layers[i].n.r_in(n[i-1], kx0_sqrt, self.k0)
                    # after layer
                    if not isinstance(self.layers[i + 1].n, Anisotropic):
                        # with next isotropic layer
                        rp[i] = self.layers[i].n.r_out(n[i+1], kx0_sqrt, self.k0)
                    else:
                        # with next anisotropic layer
                        x1 = self.layers[i].n.p_div_q(kx0_sqrt, self.k0)
                        x2 = self.layers[i+1].n.p_div_q(kx0_sqrt, self.k0)
                        rp[i] = (x2 - x1) / (x2 + x1)

            M0 = np.array([[1/tp[0], rp[0]/tp[0]], [rp[0]/tp[0], 1/tp[0]]])
            for i in range(1, len(n)-1):
                if isinstance(self.layers[i].n, Anisotropic):
                    # go through and out anisotropic layer
                    kz_pl = self.layers[i].n.kz_plus(kx0_sqrt, self.k0)
                    kz_mn = self.layers[i].n.kz_minus(kx0_sqrt, self.k0)
                    Mi = np.array([[np.exp(-1j*kz_pl*d[i])/tp[i],
                                    np.exp(-1j*kz_pl*d[i])*rp[i]/tp[i]],
                                   [np.exp(1j*kz_mn*d[i])*rp[i]/tp[i],
                                    np.exp(1j*kz_mn*d[i])/tp[i]]])
                elif isinstance(self.layers[i].n, FunctionType):
                    # Gradient layer
                    Mi = self.GradLayerMatrix(theta, n, d, i)
                else:
                    # Normal layer
                    Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/tp[i],
                                    np.exp(-1j*k_z[i]*d[i])*rp[i]/tp[i]],
                                   [np.exp(1j*k_z[i]*d[i])*rp[i]/tp[i],
                                    np.exp(1j*k_z[i]*d[i])/tp[i]]])
                M0 = M0@Mi
            if M0[0, 0] == 0:
                R = 1
            else:
                R = M0[1, 0]/M0[0, 0]

        # All layers for 's'
        else:
            if self.polarisation_ratio == 0:
                Rs = 0
            else:
                M0 = np.array([[1/ts[0], rs[0]/ts[0]], [rs[0]/ts[0], 1/ts[0]]])
                for i in range(1, len(n)-1):
                    if isinstance(self.layers[i].n, FunctionType):
                        # Gradient layer
                        Mi = self.GradLayerMatrix(theta, n, d, i)
                    else:
                        # Normal layer
                        Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/ts[i],
                                        np.exp(-1j*k_z[i]*d[i])*rs[i]/ts[i]],
                                       [np.exp(1j*k_z[i]*d[i])*rs[i]/ts[i],
                                        np.exp(1j*k_z[i]*d[i])/ts[i]]])
                    M0 = M0@Mi
                if M0[0, 0] == 0:
                    R = 1
                else:
                    R = M0[1, 0]/M0[0, 0]
        return R

    def GradLayerMatrix(self, theta, n, d, grad_num):
        """Include output but not input layer.

        Parameters.
        theta : int
            angle to calculate on.
        n : array[float]
            refractive index array from 0 to N.
        d : array[float]
            layers thickness with d[0]=d[n]=0 on semiinfinite lborder layers.
        number : int
            gradient layer number.

        Returns.
        -------
        Mtot : matrix [2,2]
            gradient layer matrix to reflection calculus.
        """
        Mtot = np.array([[1, 0], [0, 1]])
        dx = d[grad_num]/self.gradient_resolution
        kx0 = (self.k0 * n[0] * np.sin(theta))**2
        n_range = np.linspace(0, 1, self.gradient_resolution)

        ngrad = self.layers[grad_num].n(n_range)
        if self.fix_granient_n_minimum:
            for i in range(self.gradient_resolution):
                if ngrad[i] < 1:
                    ngrad[i]  = 1
        for i in range(1, self.gradient_resolution):
            ni = ngrad[i-1]
            ni1 = ngrad[i]

            ki = SM.sqrt((self.k0*ni)**2 - kx0)
            ki1 = SM.sqrt((self.k0*ni1)**2 - kx0)
            if self.polarisation == 'p':
                a = ki * ni1**2
                b = ki1 * ni**2
            else:
                a = ki
                b = ki1
            r = (a - b) / (a + b)
            t = 2*a / (a + b)
            kidx = ki*dx
            M = np.array([[np.exp(-1j*kidx) / t,
                           np.exp(-1j*kidx)*r / t],
                          [np.exp(1j*kidx)*r / t,
                           np.exp(1j*kidx) / t]])
            Mtot = Mtot@M

        # Output layer
        ni = ni1
        ni1 = n[grad_num + 1]
        ki = SM.sqrt((self.k0*ni)**2 - kx0)
        ki1 = SM.sqrt((self.k0*ni1)**2 - kx0)
        if self.polarisation == 'p':        
            r = (ki * ni1**2 - ki1 * ni**2) / (ki * ni1**2 + ki1 * ni**2)
            t = (2 * ki * ni1**2) / (ki * ni1**2 + ki1 * ni**2)
        else:
            r = (ki - ki1) / (ki + ki1)
            t = (2 * ki) / (ki + ki1)
        M = np.array([[np.exp(-1j*ki*dx)/t, np.exp(-1j*ki*dx)*r/t],
                     [np.exp(1j*ki*dx)*r/t, np.exp(1j*ki*dx)/t]])
        Mtot = Mtot@M
        return Mtot

    # -----------------------------------------------------------------------
    # --------------- secondary functions -----------------------------------
    # -----------------------------------------------------------------------

    def get_SPR_curve(self, lambda_range):
        """Get minimum R(ϴ, λ) for actual set in range.

        Parameters
        ----------
        lambda_range : range
            Range to search.

        Returns
        -------
        array
            [λ, ϴ(SPP), R(SPP)]
        """
        Rmin_curve = []
        # bnds = self.curve_angles_range

        self.k0_asylum = self.k0

        for wl in lambda_range:
            self.wavelength = wl

            # lambda func R(theta) and search of minimum
            R_dif = lambda th: np.abs(self.R_deg(th))**2
            theta_min = minimize_scalar(R_dif, bounds=self.curve_angles_range, method='Bounded')

            # minimum value
            Rw_min = np.abs(self.R_deg(theta_min.x))**2

            # bnds = (theta_min.x - 0.5, theta_min.x + 0.5)
            Rmin_curve.append([wl, theta_min.x, Rw_min])

        self.k0 = self.k0_asylum
        return np.array(Rmin_curve)

    def TIR(self):
        """Return Gives angle of total internal reflecion."""
        # initial conditions
        warning = None
        TIR_ang = 0
        if (sum(self.d) > 2.0 * self.wavelength):
            warning = 'Warning! System too thick to determine\
                total internal reflection angle explicitly!'

        # Otto scheme is when last layer is metal
        if self.n[len(self.n)-1].real < self.n[0].real:
            TIR_ang = np.arcsin(self.n[len(self.n)-1].real/self.n[0].real)
        else:
            # Kretchman scheme is when second layer is metal
            if self.n[1].real > self.n[0].real:
                warning = 'Warning! System too complicated to\
                    determine total internal reflection angle explicitly!'
            for a in range(1, len(self.n)-1):
                if self.n[a].real < self.n[0].real:
                    TIR_ang = np.arcsin(self.n[a].real /
                                        self.n[0].real)
                    break

        # If not found
        if TIR_ang == 0:
            warning = 'Warning! Total internal\
                reflection not occures in that system'
            TIR_ang = np.pi/2

        # if warnings occure
        if warning is not None:
            print(warning)

        return 180*TIR_ang/(np.pi)

    def show_info(self, show_profiles=True):
        """Show set parametrs."""
        word=' Unit parametrs '
        print(f"{word:-^30}")

        print("k0:", self.k0)
        print("λ: ", self.wavelength)
        print("n: ", self.n)
        print("d: ", self.d)
        if show_profiles:
            self.gradient_profiles(dpi=200)

    def gradient_profiles(self, dpi=None, name='Gradient layers profiles'):
        """Parameters.
    
        func : function
            form of gradient layer in [0,1].
        """
        gradient_found = False
        complex_found = False
        # print(self.layers)
        for key, value in self.layers.items():
            if isinstance(value.n, FunctionType):
                # if found first
                if gradient_found == False:
                    # Initialize plots
                    if dpi is None:
                        fig, ax = plt.subplots()
                    else:
                        fig, ax = plt.subplots(dpi=dpi)
                    ax.grid()
                    n_range = np.linspace(0, 1, self.gradient_resolution)
                    plt.title(name)
                        # remember that plot initialized
                    gradient_found = True
                # draw layer
                if value.name is None:
                    my_label = "Gradient at " + str(key)
                else:
                    my_label = value.name
                nnn = value.n(n_range)
                if self.fix_granient_n_minimum:
                    for i in range(self.gradient_resolution):
                        if nnn[i] < 1:
                            nnn[i]  = 1
                # check for complex vales
                if not complex_found:
                    for i in nnn:
                        if isinstance(i, complex):
                            complex_found = True
                if complex_found:
                    nnn = [np.real(i) for i in nnn]

                ax.plot(n_range, nnn, label=my_label)


        # final
        if gradient_found:
            plt.ylabel('n')
            plt.xlabel('d,%')
            plt.legend(loc='best')
            if not complex_found:
                plt.show()
            print("Gradient profiles shown in plots.")
        else:
            print("No gradient layers found.")
        
        # add complex if found
        if complex_found:
            # init
            n_range = np.linspace(0, 1, 200)
            # fearch for imaginary
            for key, value in self.layers.items():
                if isinstance(value.n, FunctionType):
                    complex_found = False
                    nnn = value.n(n_range)
                    # check for complex vales
                    for i in nnn:
                        if isinstance(i, complex):
                            complex_found = True
                    if complex_found:
                        nnn = [np.imag(i) for i in nnn]
                        if value.name is None:
                            my_label = "Imaginary gradient at " + str(key)
                        else:
                            my_label = value.name
                        ax.plot(n_range, nnn, label=my_label)
            plt.legend(loc='best')
            plt.show()

    def plot_SPR_curve(self, Rmin_curve, plot_2d=False, view_angle=None):
        """Plot R(ϴ, λ).

        Parameters
        ----------
        Rmin_curve : array
            [λ, ϴ(SPP), R(SPP)].
        plot_2d : bool, optional
            If plots R(ϴ) and R(λ)are shown. The default is False.
        view_angle : float, optional
            angle to rotate 3d. The default is None.
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xs=Rmin_curve[:, 0], ys=Rmin_curve[:, 1], zs=Rmin_curve[:, 2])
        ax.set_zlim3d(0, max(Rmin_curve[:, 2]))

        ax.set_xlabel('λ , nm')
        ax.set_ylabel('θ, °')
        ax.set_zlabel('R')
        if view_angle is not None:
            ax.view_init(view_angle[0], view_angle[1])

        plt.show()

        if plot_2d:
            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 1])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('θ, °')
            plt.show()

            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 2])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('R')
            plt.show()

    curve_angles_range = [50,70]

    def pointSPR(self):
        # print(self.curve_angles_range)
        theta_min = minimize_scalar(lambda x: np.abs(self.R_deg(x))**2,
                    bounds=self.curve_angles_range, method='Bounded')
        Rw_min = np.abs(self.R_deg(theta_min.x))**2
        return Rw_min, theta_min.x
        
        
        
        
def main(): print('This is library, you can\'t run it :)')


if __name__ == "__main__": main()
