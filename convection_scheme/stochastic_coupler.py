"""Class to couple downdrafts to updrafts."""

import sys
import numpy as np
from metpy.units import units

from dparcel.thermo import equivalent_potential_temperature

from stochastic_generator import StochasticThermalGenerator, equilibrate

MAX_PRECIP_DOWNDRAFT_HEIGHT = 4000*units.meter
PRECIP_LAYER_THICKNESS = 2000*units.meter


class StochasticCoupledThermalGenerator(StochasticThermalGenerator):
    """
    Calculates stochastic coupled updrafts and downdrafts.

    The stochastic model follows Romps and Kuang (2010).

    Inherits the methods and attributes of StochasticThermalGenerator,
    with the following additions.

    Methods:
        precipitation_driven: Simulate precipitation-driven downdrafts.
        overshooting: Simulate overshooting downdrafts.
        multi: Simulate precipitation-driven and overshooting downdrafts.
        ensemble: Simulate an ensemble of coupled updrafts and downdrafts.


    Attributes:
        i_min_theta_e: Index of the level of minimum theta-e in the
            sounding.

    References:
        Romps, D. M., & Kuang, Z. (2010). Nature versus nurture in
        shallow convection. Journal of the Atmospheric Sciences, 67(5),
        1655–1666. https://doi.org/10.1175/2009JAS3307.1
    """

    def __init__(self, pressure, height, temperature, specific_humidity):
        """
        Creates an instance of Coupler.

        Args:
            pressure: Increasing array of pressures at model levels.
            height: Array of corresponding heights at model levels.
            temperature: Array of corresponding temperatures.
            specific_humidity: Array of corresponding specific humidities.

        Returns:
            An instance of Coupler.
        """
        super().__init__(pressure, height, temperature, specific_humidity)

        theta_e = equivalent_potential_temperature(
            pressure, temperature, specific_humidity)
        self.i_min_theta_e = np.argmin(theta_e)

    def precipitation_driven(
            self, updraft, lambda_, sigma, drag, basic=True):
        """
        Simulate precipitation-driven downdrafts.

        Args:
            updraft: An instance of UpdraftResult corresponding to
                the associated updraft.
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            basic: Set to True to skip calculation of the detrained
                air properties.

        Returns:
            Two DowndraftResult objects: the first for a
            precipitation-driven downdraft initiating at the level
            of minimum theta-e in the sounding, and the second
            initiating at the level of maximum cumulative available
            precipitation.

            If the updraft fails to generate any precipitation,
            (None, None) will be returned. If there is no
            precipitation available at the level of minimum theta-e,
            its DowndraftResult will be replaced by None.
        """
        if np.all(updraft.precipitation == 0):
            return None, None

        total_precip = np.zeros(self.height.size)*units.dimensionless
        i_init_down_min = np.argmin(
            np.abs(self.height - MAX_PRECIP_DOWNDRAFT_HEIGHT))
        for i_init_down in range(i_init_down_min, self.height.size):
            i_precip_top = np.argmin(
                np.abs(self.height
                       - (self.height[i_init_down] + PRECIP_LAYER_THICKNESS))
            )
            total_precip[i_init_down] = np.sum(
                updraft.precipitation[i_precip_top:i_init_down+1])

        if total_precip[self.i_min_theta_e] > 0:
            theta_e_downdraft = self.precipitation_downdraft(
                self.i_min_theta_e, total_precip[self.i_min_theta_e],
                0*units.meter/units.second, lambda_, sigma, drag, basic=basic)
        else:
            theta_e_downdraft = None

        i_max_precip = np.argmax(total_precip)
        max_precip_downdraft = self.precipitation_downdraft(
            i_max_precip, total_precip[i_max_precip],
            0*units.meter/units.second, lambda_, sigma, drag, basic=basic)

        return theta_e_downdraft, max_precip_downdraft

    def overshooting(self, updraft, lambda_, sigma, drag, basic=True):
        """
        Simulate an overshooting downdraft.

        Args:
            updraft: An instance of UpdraftResult corresponding to
                the associated updraft.
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            basic: Set to True to skip calculation of the detrained
                air properties.

        Returns:
            A DowndraftResult object containing the overshooting
            downdraft properties.
        """
        i_top = np.min(np.argwhere(~np.isnan(updraft.velocity)))
        t_init_down, q_init_down, l_init_down = equilibrate(
            self.pressure[i_top], updraft.temperature[i_top],
            updraft.specific_humidity[i_top], updraft.liquid_content[i_top])

        downdraft = self.downdraft(
            i_top, t_init_down, q_init_down, l_init_down,
            0*units.meter/units.second, lambda_, sigma, drag, basic=basic)

        return downdraft

    def multi(
            self, i_init_up, t_pert, q_pert, l_initial, w_initial,
            lambda_, sigma, drag, l_crit, basic=True):
        """
        Simulate an updraft and three associated coupled downdrafts.

        Args:
            i_init_up: Index of the updraft initiation level.
            t_pert: Initial updraft temperature perturbation. The initial
                temperature is the environmental value plus t_perturb.
            q_pert: Initial updraft specific humidity perturbation. The
                inital specific humidity is the environmental value
                plus q_perturb.
            l_initial: Initial updraft liquid water content (mass of liquid
                per unit total mass).
            w_initial: Initial updraft velocity (must be non-negative).
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            l_crit: The critical liquid water content above which
                precipitation forms in the updraft.
            basic: Set to True to skip calculation of the detrained
                air properties.

        Returns:
            An UpdraftResult object containing the updraft properties
            and three DowndraftResult objects: the first for a
            precipitation-driven downdraft initiating at the level
            of minimum theta-e in the sounding, the second for a
            precipitation-driven downdraft initiating at the level of
            maximum cumulative available precipitation, and the third
            for an overshooting downdraft.
        """
        updraft = self.updraft(
            i_init_up, t_pert, q_pert, l_initial, w_initial,
            lambda_, sigma, drag, l_crit, basic=basic)
        downdraft1, downdraft2 = self.precipitation_driven(
            updraft, lambda_, sigma, drag, basic=basic)
        downdraft3 = self.overshooting(
            updraft, lambda_, sigma, drag, basic=basic)

        return updraft, downdraft1, downdraft2, downdraft3

    def ensemble(
            self, i_init_up, t_pert, q_pert, l_initial, w_initial,
            lambda_, sigma, drag, l_crit, n_runs, basic=True):
        """
        Simulate an ensemble of coupled updrafts and downdrafts.

        The ensemble has a spectrum of entrainment rates, with identical
        initial conditions.

        Args:
            i_init_up: Index of the updraft initiation level.
            t_pert: Initial updraft temperature perturbation. The initial
                temperature is the environmental value plus t_perturb.
            q_pert: Initial updraft specific humidity perturbation. The
                inital specific humidity is the environmental value
                plus q_perturb.
            l_initial: Initial updraft liquid water content (mass of liquid
                per unit total mass).
            w_initial: Initial updraft velocity (must be non-negative).
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            l_crit: The critical liquid water content above which
                precipitation forms in the updraft.
            n_runs: Number of calculations in the ensemble.
            basic: Set to True to skip calculation of the detrained
                air properties.

        Returns:
            Four arrays of length rates.size, one containing
            UpdraftResult instances and three containing DowndraftResult
            instances. The contents of the arrays correspond to the
            four outputs of CoupledThermalGenerator.multi.
        """
        updrafts = np.empty(n_runs, dtype='object')
        downdrafts1 = np.empty(n_runs, dtype='object')
        downdrafts2 = np.empty(n_runs, dtype='object')
        downdrafts3 = np.empty(n_runs, dtype='object')

        n_done = 0
        for i in range(n_runs):
            n_done += 1
            sys.stdout.write(f'\rCalculation {n_done} of {n_runs}    ')
            (updrafts[i], downdrafts1[i],
             downdrafts2[i], downdrafts3[i]) = self.multi(
                 i_init_up, t_pert, q_pert, l_initial, w_initial,
                 lambda_, sigma, drag, l_crit, basic=basic)

        sys.stdout.write('\n')
        return updrafts, downdrafts1, downdrafts2, downdrafts3
