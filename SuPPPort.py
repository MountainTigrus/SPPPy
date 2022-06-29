# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:24:08 2022.

Additional type classes and procedures for SPPPy

@author: THzLab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy.lib import scimath as SM
from scipy.optimize import minimize, minimize_scalar
# from sympy import *
from scipy import interpolate
from types import FunctionType

class Layer:
    """Container fpr layer parametrs"""

    def __init__(self, n, thickness, name=None):
        """New layer.

        Parameters
        ----------
        n : float, complex or type
            tepe for layer.
        thickness : float
            thickness of a layer.
        name : strinf, optional
            name of a layer. The default is None.
        """
        self.n = n
        self.thickness = thickness
        self.name = name
        
    def __repr__(self):
        """Magic representation."""
        return '\n - Layer: ' + str(self.n) + ', with d ' + str(self.thickness) + "\n"

class LorentzDrude:
    """Metall layer with Lorentz-Drude model of material permittivity."""

    def __init__(self, wp, wt, name=None):
        """Create new metall with complex refractive index.

        Parameters
        ----------
        wp : float
            The plasma frequency
        wt : float
            The collision frequency or damping factor
        name: string
            Material name
        """
        self.name = name
        self.refractivity = lambda lam:\
            SM.sqrt(1 - wp**2/((2*np.pi*3e8/lam)**2 + 1j*wt*(2*np.pi*3e8/lam)))

    def CRI(self, wavelength):
        return self.refractivity(wavelength)
    
    def show_CRI(self, lambda_range):
        """Plot complex refractive index.

        Parameters
        ----------
        lambda_range : array
            Range. The default is None - all data.
        """
        nnn = []
        kkk = []
        for lam in lambda_range:
            ref = self.refractivity(lam)
            nnn.append(np.real(ref))
            kkk.append(np.imag(ref))
            
        fig, ax = plt.subplots()
        ax.grid()
        rang = np.around(lambda_range*1e6, 3)
        ax.plot(rang, nnn, label='n')
        ax.plot(rang, kkk, label='k')
        if self.name==None:
            plt.title(f'Material omplex refractive index')
        else:
            plt.title(f'Complex refractive index of {self.name}')
        plt.legend(loc='best')
        plt.ylabel('Value')
        plt.xlabel('Wavelength, µm')
        plt.show()
    
    

class MaterialDispersion:
    """Metall layer with complex refractive index."""

    def __init__(self, material, base_file=None):
        """Create new metall with complex refractive index.

        Parameters
        ----------
        metall : string
            Name of a metall in base.
        base_file : string, optional
            Path to not default file. The default is None.
        """
        if base_file is None:
            self.base_file = "SPPPy/PermittivitiesBase.csv"
        else:
            self.base_file = base_file
        self.name = material

        # Dig for a data
        Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        if not (self.name in Refraction_data['Element'].to_list()):
            print(f"Element {self.name} not found!")
            return

        Refraction_data = Refraction_data[Refraction_data["Element"] == self.name][[
            "Wavelength", "n", "k"]].to_numpy()

        # Get scope of definition
        self.min_lam = Refraction_data[0][0]
        self.max_lam = Refraction_data[-1][0]

        self.n_func = interpolate.interp1d(
            Refraction_data[:, 0], Refraction_data[:, 1])
        self.k_func = interpolate.interp1d(
            Refraction_data[:, 0], Refraction_data[:, 2])

    # -------------------- work with selected material -----------------------
    def __repr__(self):
        """Magic representation."""
        return "dispersion, \"" + self.name + "\""

    def CRI(self, wavelength):
        """Take complex refractive index.

        Parameters
        ----------
        wavelength : float
            wavelength.

        Returns
        -------
        complex
            Complex refractive index on given wavelength.
        """
        if wavelength*1e6 >= self.min_lam and wavelength*1e6 <= self.max_lam:
            return (self.n_func(wavelength*1e6) + self.k_func(wavelength*1e6)*1j)
        else:
            print("Wavelength is out of bounds!")
            print(f"CRI for {self.name} defined in: [{self.min_lam},{self.max_lam}]µm, and given: {wavelength*1e6}µm")
            if wavelength*1e6 <= self.min_lam:
                return (self.n_func(self.min_lam) + self.k_func(self.min_lam))
            if wavelength*1e6 >= self.max_lam:
                return (self.n_func(self.max_lam) + self.k_func(self.max_lam))                       

    def show_CRI(self, lambda_range=None):
        """Plot complex refractive index.

        Parameters
        ----------
        lambda_range : array
            Range. The default is None - all data.
        """
        fig, ax = plt.subplots()
        ax.grid()

        must_cut = False
        n_range = []

        if lambda_range is not None:
            if lambda_range[0] < self.min_lam * 1e-6:
                print(f"Minimal bound ({lambda_range[0]*1e6}) is out of range ({self.min_lam} µm)")
                must_cut = True
            if lambda_range[-1] > self.max_lam* 1e-6:
                print(f"Minimal bound ({lambda_range[-1]*1e6}) is out of range ({self.max_lam} µm)")
                must_cut = True
            
            if must_cut:
                i = 0
                while i < len(lambda_range):

                    if not(lambda_range[i]>self.max_lam*1e-6 or lambda_range[i]<self.min_lam*1e-6):
                        n_range.append(lambda_range[i]*1e6)
                    i += 1

                if len(n_range)==0:
                    n_range = np.linspace(self.min_lam, self.max_lam, 500)
            else: n_range = lambda_range*1e6
        else: n_range = np.linspace(self.min_lam, self.max_lam, 500)

        # print(n_range)
        nnn = [self.n_func(j) for j in n_range]
        kkk = [self.k_func(j) for j in n_range]

        ax.plot(n_range, nnn, label='n')
        ax.plot(n_range, kkk, label='k')
        plt.title(f'Complex refractive index of {self.name}')
        plt.legend(loc='best')
        plt.ylabel('Value')
        plt.xlabel('Wavelength, µm')
        plt.show()

    # -------------------- work with base material ---------------------------

    def add_material(self, element, material_file):
        """Add new material from file.

        Parameters
        ----------
        element : str
            element name.
        material_file : file csv from refractiveindex.info
            parametrs.
        """
        # open base file
        Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        # print(Refraction_data)
        if (element in Refraction_data['Element'].to_list()):
            print("Element already exist! Create new one and use 'merge_materials'")
            return

        # open file with new data, pack csv from refractiveindex.info
        b = []
        a = np.loadtxt(material_file, dtype=str, skiprows=1, delimiter=",")
        for i in a[:]:
            b.append([float(i[0] + '.' + i[1]), float(i[2] +
                     '.' + i[3]), float(i[4] + '.' + i[5])])
        New_df = pd.DataFrame(b, columns=["Wavelength", "n", "k"])
        New_df["Element"] = [element]*New_df.shape[0]

        Refraction_data = Refraction_data.append(New_df, ignore_index=True)
        Refraction_data = Refraction_data.sort_values(["Element", "Wavelength"], ascending=[True, True])
        Refraction_data = Refraction_data.reindex() # [["Element", "Wavelength", "n", "k"]]
        Refraction_data.to_csv(self.base_file)

    def delete_material(self, material):
        """Remove metall from base.

        Parameters
        ----------
        metall : string
            Metall name.
        """
        Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)

        for a, b in Refraction_data.iterrows():
            # a is index, b is row !!!
            if b['Element'] == material:
                Refraction_data = Refraction_data.drop(a)

        Refraction_data = Refraction_data.sort_values(["Element", "Wavelength"], ascending=[True, True])
        Refraction_data = Refraction_data.reindex() # [["Element", "Wavelength", "n", "k"]]
        Refraction_data.to_csv(self.base_file)

    def merge_materials(self, primary, second, new_name, delete_origin=True):
        """Merge two materials from base.
        If wavelength of second will get in wavelength of primary it will cut

        Parameters
        ----------
        primary : string
            primary material.
        second : string
            second material.
        new_name : string
            new name.
        """
        Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        if (new_name in Refraction_data['Element'].to_list()):
            if not (delete_origin and (new_name == second or new_name == primary)):
                print("Can't save to material that already exist!")
                return

        if not (primary in Refraction_data['Element'].to_list()):
            print(f"{primary} not exist, nothing to merge!")
            return          
        if not (second in Refraction_data['Element'].to_list()):
            print(f"{second} not exist, nothing to merge!")
            return                     
        
        # take primary
        primary_base = Refraction_data[Refraction_data['Element']==primary][["Wavelength", "n", "k"]]
        primary_min = primary_base["Wavelength"].min()
        primary_max = primary_base["Wavelength"].max()

        # take seconf
        second_base = Refraction_data[Refraction_data['Element']==second][["Wavelength", "n", "k"]]

        # cut second base
        second_base = pd.concat([second_base[second_base["Wavelength"]<primary_min],
               second_base[second_base["Wavelength"]>primary_max]], ignore_index = True)

        # merge bases
        primary_base = pd.concat([primary_base, second_base], ignore_index = True)
        primary_base["Element"] = [new_name]*primary_base.shape[0]

        # remove old materials
        if delete_origin:
            for a, b in Refraction_data.iterrows():
                if b["Element"] == primary or b["Element"] == second:
                    Refraction_data = Refraction_data.drop(a)

        Refraction_data = pd.concat([Refraction_data, primary_base], ignore_index = True)
        
        Refraction_data = Refraction_data.sort_values(["Element", "Wavelength"], ascending=[True, True])
        Refraction_data = Refraction_data.reindex() # [["Element", "Wavelength", "n", "k"]]
        Refraction_data.to_csv(self.base_file)

    def show_base_info(self):
        print(f"File: {self.base_file}")
        Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        print(f"Shape: {Refraction_data.shape}")
        print(f"Columns: {Refraction_data.columns}")
        print(f"Index: {Refraction_data.index}")

    def materials_List(self):
        """Show all metals with definition range in wavelength.

        Parameters
        ----------
        base_file : string, optional
            Path to not default file. The default is None.
        """

        Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        agg_func_selection = {'Wavelength': ['min', 'max']}
        print(Refraction_data.sort_values(["Element", "Wavelength"], ascending=[True,
                         True]).groupby(['Element']).agg(agg_func_selection))


class Anisotropic:
    """Anisotropic dielectric layer."""

    def __init__(self, n0, n1, main_angle):
        """Anisotropic layer.

        Parameters
        ----------
        n0 : float
            ordinary reflection_data coeficient.
        n1 : float
            extraordinary reflection_data coeficient.
        main_angle : float
            Principle axis angle in degree
        """
        self.n0 = n0
        self.n1 = n1
        self.main_angle = np.pi * main_angle / 180

        # Equivalent rafractive indices
        self.ny_2 = (n0 * np.cos(self.main_angle))**2 + (n1 * np.sin(self.main_angle))**2
        self.nz_2 = (n0 * np.sin(self.main_angle))**2 + (n1 * np.cos(self.main_angle))**2
        self.nyz = (n0**2 - n1**2) * np.sin(self.main_angle) * np.cos(self.main_angle)

        # wave vector blocks
        self.kz_dot = lambda beta, k0: SM.sqrt(k0**2 * self.ny_2 -
                          beta**2 * self.ny_2 / self.nz_2)
        self.K = SM.sqrt(1 - self.nyz**2 / (self.ny_2 * self.nz_2))
        self.deltaK = lambda beta: (beta * self.nyz) / self.nz_2

    def kz_plus(self, beta, k0):
        """Kz+."""
        return self.kz_dot(beta, k0) * self.K + self.deltaK(beta)

    def kz_minus(self, beta, k0):
        """Kz-."""
        return self.kz_dot(beta, k0) * self.K - self.deltaK(beta)

    def r_in(self, n_prev, beta, k0):
        """r01."""
        a = n_prev**2 / SM.sqrt(n_prev**2 - (beta/k0)**2)
        b = self.p_div_q(beta, k0)
        return - (a - b) / (a + b)

    def r_out(self, n_next, beta, k0):
        """r12."""
        a = self.p_div_q(beta, k0)
        b = n_next**2 / SM.sqrt(n_next**2 - (beta/k0)**2)
        return - (a - b) / (a + b)

    def p_div_q(self, beta, k0):
        """p/q for rij."""
        return SM.sqrt(self.ny_2 * self.nz_2) * self.K / SM.sqrt(self.nz_2 - (beta/k0)**2)

    def __repr__(self):
        """Magic representation."""
        return "anisotropic, n=(" + str(self.n0) + ", " + str(self.n1) + "), angle=" + str(180*self.main_angle/np.pi)


# ------------------------------------------------------------------------
# ----------------------- Other functions --------------------------------
# ------------------------------------------------------------------------


def gradient_profile(func, name=None, dpi=None):
    """Parameters.

    func : function
        form of gradient layer in [0,1].
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    n_range = np.linspace(0, 1, 50)
    nnn = func(n_range)
    ax.plot(n_range, nnn)
    if name is None:
        plt.title('Gradient layer profile Shape')
    else:
        plt.title(name)
    plt.ylabel('n')
    plt.xlabel('d,%')
    plt.show()


def plot_graph(x, y, name='Reflection data', tir_ang=None, label=None,
               dpi=None, ox_label='ϴ', oy_label='R', oy_limits=True):
    """Parameters.

    x : array(float)
        x cordinates.
    y : array(float)
        x cordinates.
    name : string
        plot name..
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if label is not None:
        ax.plot(x, y, label=label)
        plt.legend(loc='best')
    else:
        ax.plot(x, y)
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.title(name)
    if oy_limits:
        plt.ylim([0, 1.05])
    plt.ylabel(oy_label)
    plt.xlabel(ox_label)
    plt.show()


def plot_graph_complex(R_array, name='Reflection data', fix_limits=False, dpi=None,):
    """Parameters.

    x : array(float)
        x cordinates.
    y : array(float)
        x cordinates.
    name : string
        plot name..
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    
    x = []
    y = []
    for i in range(0, len(R_array)):
        x.append(np.real(R_array[i]))
        y.append(np.imag(R_array[i]))

    t = np.linspace(0, 2 * np.pi, len(R_array))
    plt.scatter(t,x,c=y)
    plt.colorbar()

    if fix_limits:
        plt.ylim([-1.05, 1.05])
        # plt.xlim([-1.05, 1.05])

    plt.title(name)

    plt.ylabel("Re(R)")
    plt.xlabel("Im(R)")
    plt.show()


def plot_2d(x, y, name='Plot', label=None, dpi=None):
    """Parameters.

    x : array(float)
        x cordinates.
    y : array(float)
        x cordinates.
    name : string
        plot name.
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if label is not None:
        ax.plot(x, y,label=label)
        plt.legend(loc='best')
    else:
        ax.plot(x, y)
    plt.title(name)

    plt.ylabel('R')
    plt.xlabel('ϴ°')
    plt.show()


def multiplot_graph(plots, name='Plot', tir_ang=None, dpi=None):
    """Parameters.

    plots : array(x, y, name)
        like in "Plot_Graph".
    name : string
        plot name.
    tir_ang : int, optional
        Total internal reflection_data angle. The default is None.
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if len(plots[0]) == 3:
        for i in plots:
            ax.plot(i[0], i[1], label=i[2])
    elif len(plots[0]) == 4:
        for i in plots:
            ax.plot(i[0], i[1], label=i[2], linestyle=i[3])
    else:
        print('Not valid array dimension')

    plt.legend(loc='best')
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.ylabel('R')
    plt.xlabel('ϴ°')
    plt.title(name)
    plt.show()


def profile_analyzer(theta_range, reflection_data):
    """Find SPP angle and dispersion halfwidth.

    Parameters.
    reflection_data : array[float]
        reflection_data profile.
    theta_range : range(start, end, seps)
        Range of function definition.

    Returns.
    -------
    xSPPdeg : float
        SPP angle in grad.
    halfwidth : float
        halfwidth.
    """
    div_val = (theta_range.max() - theta_range.min())/len(reflection_data)

    # minimum point - SPP
    yMin = min(reflection_data)
    # print('y_min ',yMin)
    xMin,  = np.where(reflection_data == yMin)[0]
    # print('x_min ',xMin)
    xSPPdeg = theta_range.min() + div_val * xMin

    # first maximum before the SPP
    Left_Part = reflection_data[0:xMin]
    if len(Left_Part) > 0:
        yMax = max(Left_Part)
    else:
        yMax = 1
    left_border = 0
    right_border = reflection_data
    half_height = (yMax-yMin)/2
    point = xMin
    while (reflection_data[point] < yMin + half_height):
        point -= 1
    left_border = point
    # print('left hw ', left_border)
    point = xMin
    while (reflection_data[point] < yMin + half_height):
        point += 1
    right_border = point
    # print('rigth hw ', right_border)

    halfwidth = div_val * (right_border - left_border)

    # print('xSPPdeg = ', xSPPdeg, 'halfwidth ', halfwidth)
    return xSPPdeg,  halfwidth

