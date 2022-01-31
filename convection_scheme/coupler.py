"""Class to couple downdrafts to updrafts."""

import sys
import numpy as np
from metpy.units import units

from dparcel.thermo import equivalent_potential_temperature

from thermal_generator import ThermalGenerator, equilibrate

MAX_PRECIP_DOWNDRAFT_HEIGHT = 4000*units.meter
PRECIP_LAYER_THICKNESS = 2000*units.meter


class CoupledThermalGenerator(ThermalGenerator):
    """
    Calculates coupled updrafts and downdrafts.

    Inherits the methods and attributes of ThermalGenerator, with
    the following additions.

    Methods:
        precipitation_driven: Simulate precipitation-driven downdrafts.
        overshooting: Simulate overshooting downdrafts.
        multi: Simulate precipitation-driven and overshooting downdrafts.
        ensemble: Simulate an ensemble of coupled updrafts and downdrafts.


    Attributes:
        i_min_theta_e: Index of the level of minimum theta-e in the sounding.
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

    def precipitation_driven(self, updraft, rate, dnu_db, drag, basic=True):
        """
        Simulate precipitation-driven downdrafts.

        Args:
            updraft: An instance of UpdraftResult corresponding to
                the associated updraft.
            rate: Entrainment rate (should have dimensions
                of 1/length).
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2.
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
                0*units.meter/units.second, rate, dnu_db, drag, basic=basic)
        else:
            theta_e_downdraft = None

        i_max_precip = np.argmax(total_precip)
        max_precip_downdraft = self.precipitation_downdraft(
            i_max_precip, total_precip[i_max_precip],
            0*units.meter/units.second, rate, dnu_db, drag, basic=basic)

        return theta_e_downdraft, max_precip_downdraft

    def overshooting(self, updraft, rate, dnu_db, drag, basic=True):
        """
        Simulate an overshooting downdraft.

        Args:
            updraft: An instance of UpdraftResult corresponding to
                the associated updraft.
            rate: Entrainment rate (should have dimensions
                of 1/length).
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2.
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
            0*units.meter/units.second, rate, dnu_db, drag, basic=basic)

        return downdraft

    def multi(
            self, i_init_up, t_pert, q_pert, l_initial, w_initial,
            rate_up, rate_down, dnu_db, drag, l_crit, basic=True):
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
            rate_up: Updraft entrainment rate (should have dimensions
                of 1/length).
            rate_down: Downdraft entrainment rate (should have dimensions
                of 1/length).
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2. Assumed to be the same for
                updraft and downdraft.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length. Assumed to be the
                same for updraft and downdraft.
            l_crit: The critical liquid water content above which
                precipitation forms in the updraft.
            basic: Set to True to skip calculation of the detrained
                air properties for both updraft and downdraft.

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
            rate_up, dnu_db, drag, l_crit, basic=basic)
        downdraft1, downdraft2 = self.precipitation_driven(
            updraft, rate_down, dnu_db, drag, basic=basic)
        downdraft3 = self.overshooting(
            updraft, rate_down, dnu_db, drag, basic=basic)

        return updraft, downdraft1, downdraft2, downdraft3

    def ensemble(
            self, i_init_up, t_pert, q_pert, l_initial, w_initial,
            rates_up, rates_down, dnu_db, drag, l_crit, basic=True):
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
            rates_up: Array of updraft entrainment rates (should have
                dimensions of 1/length).
            rates_down: Array of downdraft entrainment rates (should
                have dimensions of 1/length). Must have the same size
                as rates_up
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2. Assumed to be the same for
                all updrafts and downdrafts.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length. Assumed to be the
                same for all updrafts and downdrafts.
            l_crit: The critical liquid water content above which
                precipitation forms in the updraft. Assumed to be the
                same for all updrafts.
            basic: Set to True to skip calculation of the detrained
                air properties for both updrafts and downdrafts.

        Returns:
            Four arrays of length rates.size, one containing
            UpdraftResult instances and three containing DowndraftResult
            instances. The contents of the arrays correspond to the
            four outputs of CoupledThermalGenerator.multi.
        """
        rates_up = np.atleast_1d(rates_up)
        rates_down = np.atleast_1d(rates_down)
        if rates_up.size != rates_down.size:
            raise ValueError(
                'rates_up must have the same size as rates_down.')
        updrafts = np.empty(rates_up.size, dtype='object')
        downdrafts1 = np.empty(rates_up.size, dtype='object')
        downdrafts2 = np.empty(rates_up.size, dtype='object')
        downdrafts3 = np.empty(rates_up.size, dtype='object')

        n_done = 0
        for i in range(rates_up.size):
            n_done += 1
            sys.stdout.write(f'\rCalculation {n_done} of {rates_up.size}    ')
            (updrafts[i], downdrafts1[i],
             downdrafts2[i], downdrafts3[i]) = self.multi(
                 i_init_up, t_pert, q_pert, l_initial, w_initial,
                 rates_up[i], rates_down[i], dnu_db, drag, l_crit, basic=basic)
        
        sys.stdout.write('\n')
        return updrafts, downdrafts1, downdrafts2, downdrafts3
