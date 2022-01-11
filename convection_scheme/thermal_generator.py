"""Class for updraft and downdraft calculations."""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from metpy.units import units, concatenate
import metpy.calc as mpcalc
import metpy.constants as const

from pint.errors import OffsetUnitCalculusError

from dparcel.thermo import (lcl_romps, moist_lapse, wetbulb,
                            equivalent_potential_temperature,
                            saturation_specific_humidity)


class ThermalGenerator:
    """
    Collects functions for updrafts and downdrafts in a given sounding.

    Methods:
        updraft: Calculate the properties of an ascending thermal.
        downdraft: Calculate the properties of a descending thermal.

    Attributes:
        pressure: Array of pressure levels in the sounding.
        temperature: Temperature profile in the sounding.
        specific_humidity: Specific humidity profile in the sounding.
        t_virtual: Virtual temperature profile in the sounding.
        t_disp: Array with shape (pressure.size, pressure.size), where
            element (i,j) is the temperature attained by an environmental
            parcel moving adiabatically without entrainment from level i
            to level j.
        q_disp: Specific humidity array with the same structure as
            t_disp.
        l_disp: Liquid water content array with the same structure as
            t_disp.
        b_disp: Buoyancy array with the same structure as t_disp.
    """

    def __init__(self, pressure, height, temperature, specific_humidity):
        """
        Creates an instance of ThermalGenerator.

        Args:
            pressure: Increasing array of pressures at model levels.
            height: Array of corresponding heights at model levels.
            temperature: Array of corresponding temperatures.
            specific_humidity: Array of corresponding specific humidities.

        Returns:
            An instance of ThermalGenerator.
        """
        if np.any(pressure[:-1] >= pressure[1:]):
            raise ValueError('Pressure array must be strictly increasing.')

        # Find the LCL using the exact solution of Romps (2017)
        p_lcl, t_lcl = lcl_romps(pressure, temperature, specific_humidity)
        lcl_index = np.searchsorted(pressure, p_lcl)

        # initialise empty arrays
        t_disp = np.zeros((pressure.size, pressure.size))*units.kelvin
        q_disp = np.zeros((pressure.size, pressure.size))*units.dimensionless
        l_disp = np.zeros((pressure.size, pressure.size))*units.dimensionless

        for i in range(pressure.size):
            if lcl_index[i] < pressure.size:
                # temperatures for j >= lcl_index[i], i.e., below the LCL,
                # are the dry adiabatic values
                t_disp[i,lcl_index[i]:] = mpcalc.dry_lapse(
                    pressure[lcl_index[i]:], temperature[i],
                    reference_pressure=pressure[i])
                # specific humidity is constant below the LCL and equal
                # to the initial environmental value
                q_disp[i,lcl_index[i]:] = specific_humidity[i]
            if lcl_index[i] > 0:
                # temperatures for j < lcl_index[i], i.e., above the LCL,
                # are the moist pseudoadiabatic values, starting from
                # the known pressure and temperature at the LCL
                t_disp[i,:lcl_index[i]] = moist_lapse(
                    pressure[:lcl_index[i]], t_lcl[i],
                    reference_pressure=p_lcl[i], method='fast')
                # the specific humidity is the saturation value
                q_disp[i,:lcl_index[i]] = saturation_specific_humidity(
                    pressure[:lcl_index[i]], t_disp[i,:lcl_index[i]])
            # assuming total water is conserved, the final specific
            # humidity plus liquid water content equals the initial
            # specific humidity
            l_disp[i,:] = specific_humidity[i] - q_disp[i,:]

        # calculate virtual temperatures and buoyancy
        r_disp = mpcalc.mixing_ratio_from_specific_humidity(q_disp)
        tv_disp = mpcalc.virtual_temperature(t_disp, r_disp)
        r_env = mpcalc.mixing_ratio_from_specific_humidity(specific_humidity)
        tv_env = mpcalc.virtual_temperature(temperature, r_env)
        # note that the parcel arrays are 2D while the environment array
        # is 1D: the calculation is performed row-by-row on the 2D arrays,
        # not column-by-column
        b_disp = (tv_disp - tv_env)/tv_env*const.g

        self.pressure = pressure
        self.height = height
        self.temperature = temperature
        self.specific_humidity = specific_humidity
        self.t_virtual = tv_env
        self.t_disp = t_disp
        self.q_disp = q_disp
        self.l_disp = l_disp
        self.b_disp = b_disp

    def updraft(
            self, i_init, t_perturb, q_perturb, l_initial, w_initial,
            entrainment_rate, dnu_db, drag, l_crit):
        """
        Calculate the properties associated with an ascending thermal.

        Args:
            i_init: Index of the initiation level.
            t_perturb: Initial temperature perturbation. The initial
                temperature is the environmental value plus t_perturb.
            q_perturb: Initial specific humidity perturbation. The
                inital specific humidity is the environmental value
                plus q_perturb.
            l_initial: Initial liquid water content (mass of liquid
                per unit total mass).
            w_initial: Initial velocity (must be non-negative).
            entrainment_rate: Entrainment rate (should have dimensions
                of 1/length).
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            l_crit: The critical liquid water content above which
                precipitation forms.

        Returns:
            Bunch object with the folliwing fields defined --
                - **temperature** -- Array containing the parcel's
                  temperature at each level.
                - **specific_humidity** -- Array containing the parcel's
                  specific humidity at each level.
                - **liquid_content** -- Array containing the parcel's
                  liquid content at each level.
                - **precipitation** -- Array containing the mass of
                  liquid water precipitated out at each level, as a
                  fraction of the parcel mass at the same level.
                - **buoyancy** -- Array containing the parcel's buoyancy
                  at each level.
                - **velocity** -- Array containing the parcel's
                  vertical velocity at each level.
                - **m_detrained** -- Array containing the mass
                  detrained at each level, as a fraction of the original
                  mass.
                - **m_remaining** -- Array containing the mass
                  remaining at each level, as a fraction of the original
                  mass.
                - **t_detrained** -- Array containing the temperature of
                  detrained air at each level.
                - **q_detrained** -- Array containing the specific
                  humidity of detrained air at each level.
                - **l_detrained** -- Array containing the liquid
                  content of detrained air at each level.
        """
        try:
            t_initial = self.temperature[i_init] + t_perturb
        except OffsetUnitCalculusError as e:
            raise ValueError(
                f'The units of t_perturb ({t_perturb.units}) are not'
                'suitable for addition to the environmental temperature.'
                'Try kelvin or delta_degC instead.') from e
        (temperature, specific_humidity, liquid_content,
         buoyancy, precipitation) = (
            self._updraft_properties(
                i_init, t_initial, self.specific_humidity[i_init] + q_perturb,
                l_initial, entrainment_rate, l_crit)
        )

        velocity = self._velocity_profile(
            i_init, w_initial, buoyancy, drag, kind='up')
        m_detrained, m_remaining = self._detrained_mass(
            velocity, buoyancy, dnu_db, entrainment_rate, kind='up')

        t_detrained = np.full(self.pressure.size, np.nan)*units.kelvin
        q_detrained = np.full(self.pressure.size, np.nan)*units.dimensionless
        l_detrained = np.full(self.pressure.size, np.nan)*units.dimensionless
        for i in np.argwhere(~np.isnan(velocity)):
            # the components of the detrained air mix and
            # come into phase equilibrium
            t_detrained[i], q_detrained[i], l_detrained[i] = equilibrate(
                self.pressure[i], temperature[i],
                specific_humidity[i], liquid_content[i])

        result = UpdraftResult()
        result.temperature = temperature
        result.specific_humidity = specific_humidity
        result.liquid_content = liquid_content
        result.precipitation = precipitation
        result.buoyancy = buoyancy
        result.velocity = velocity
        result.m_detrained = m_detrained.to(units.dimensionless)
        result.m_remaining = m_remaining.to(units.dimensionless)
        result.t_detrained = t_detrained
        result.q_detrained = q_detrained
        result.l_detrained = l_detrained
        return result

    def downdraft(
            self, i_init, delta_Q, w_initial, entrainment_rate, dnu_db, drag):
        """
        Calculate the properties associated with a descending thermal.

        The descent is assumed to be triggered by the evaporation of
        precipitation into an environmental parcel.

        Args:
            i_init: Index of the initiation level.
            delta_Q: Total mass of liquid water initially evaporated
                into the environmental parcel at i_init.
            w_initial: Initial velocity (must be non-negative).
            entrainment_rate: Entrainment rate (should have dimensions
                of 1/length).
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.

        Returns:
            Bunch object with the folliwing fields defined --
                - **temperature** -- Array containing the parcel's
                  temperature at each level.
                - **specific_humidity** -- Array containing the parcel's
                  specific humidity at each level.
                - **liquid_content** -- Array containing the parcel's
                  liquid content at each level.
                  liquid water precipitated out at each level, as a
                  fraction of the parcel mass at the same level.
                - **buoyancy** -- Array containing the parcel's buoyancy
                  at each level.
                - **velocity** -- Array containing the parcel's
                  vertical velocity at each level.
                - **m_detrained** -- Array containing the mass
                  detrained at each level, as a fraction of the original
                  mass.
                - **m_remaining** -- Array containing the mass
                  remaining at each level, as a fraction of the original
                  mass.
                - **t_detrained** -- Array containing the temperature of
                  detrained air at each level.
                - **q_detrained** -- Array containing the specific
                  humidity of detrained air at each level.
                - **l_detrained** -- Array containing the liquid
                  content of detrained air at each level.
        """
        # find the temperature of the environmental parcel after
        # evaporation of an amount delta_Q of liquid water
        t_initial, q_initial, l_initial = equilibrate(
            self.pressure[i_init], self.temperature[i_init],
            self.specific_humidity[i_init], delta_Q)

        temperature, specific_humidity, liquid_content, buoyancy = (
            self._downdraft_properties(
                i_init, t_initial, q_initial, l_initial, entrainment_rate)
        )
        velocity = self._velocity_profile(
            i_init, w_initial, buoyancy, drag, kind='down')
        m_detrained, m_remaining = self._detrained_mass(
            velocity, buoyancy, dnu_db, entrainment_rate, kind='down')

        t_detrained = np.full(self.pressure.size, np.nan)*units.kelvin
        q_detrained = np.full(self.pressure.size, np.nan)*units.dimensionless
        l_detrained = np.full(self.pressure.size, np.nan)*units.dimensionless
        for i in np.argwhere(~np.isnan(velocity)):
            # the components of the detrained air mix and
            # come into phase equilibrium
            t_detrained[i], q_detrained[i], l_detrained[i] = equilibrate(
                self.pressure[i], temperature[i],
                specific_humidity[i], liquid_content[i])

        result = DowndraftResult()
        result.temperature = temperature
        result.specific_humidity = specific_humidity
        result.liquid_content = liquid_content
        result.buoyancy = buoyancy
        result.velocity = velocity
        result.m_detrained = m_detrained.to(units.dimensionless)
        result.m_remaining = m_remaining.to(units.dimensionless)
        result.t_detrained = t_detrained
        result.q_detrained = q_detrained
        result.l_detrained = l_detrained
        return result

    def _transition_point(self, p_initial, t_initial, q_initial, l_initial):
        """
        Finds the transition point between moist and dry descent.

        Only applies to non-entraining parcels.
        For updraft parcels, this is equivalent to an LCL calculation.

        Args:
            p_initial: Starting pressure of the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.

        Returns:
            The pressure at which the liquid water ratio in the parcel
            becomes zero, and its temperature at that point.

        References:
            Romps, DM 2017, ‘Exact Expression for the Lifting Condensation
            Level’, Journal of the atmospheric sciences, vol. 74,
            no. 12, pp. 3891–3900.
        """
        if l_initial <= 0:
            # Romps (2017) LCL calculation is valid
            return lcl_romps(p_initial, t_initial, q_initial)

        # evaluate moist adiabatic values on a coarse array between
        # starting point and the surface in order to bracket the
        # transition point
        p_check = np.arange(
            p_initial.m_as(units.mbar), self.pressure[-1].m_as(units.mbar), 10
        )*units.mbar
        p_check = concatenate([p_check, self.pressure[-1]])
        t_moist = moist_lapse(
            p_check, t_initial, reference_pressure=p_initial, method='fast')
        l_moist = (
            q_initial + l_initial
            - saturation_specific_humidity(p_check, t_moist)
        )

        if l_moist[-1] > 0:
            # moist descent only
            return self.pressure[-1], t_moist[-1]

        # now find the transition point where l == 0
        if np.any(l_moist == 0):
            # check if we have already found the transition point
            return p_check[l_moist == 0][0], t_moist[l_moist == 0][0]

        # choose a suitable bracketing interval for the transition point.
        # out of the levels that give positive l_moist, use the one
        # that gives the smallest l_moist as one end of the interval
        guess_above = p_check[
            np.nanargmin(np.where(l_moist < 0, np.nan, l_moist))
        ]
        # out of the levels that give negative l_moist, use the one
        # that gives the largest l_moist as the other end
        guess_below = p_check[
            np.nanargmax(np.where(l_moist > 0, np.nan, l_moist))
        ]

        # evaluate the parcel properties on a finely spaced array within
        # the bracketing interval and interpolate to find the point of
        # l_moist == 0
        p_check = np.linspace(
            guess_above.m_as(units.mbar), guess_below.m_as(units.mbar), 100
        )*units.mbar
        t_moist = moist_lapse(
            p_check, t_initial, reference_pressure=p_initial, method='fast')
        l_moist = (
            q_initial + l_initial
            - saturation_specific_humidity(p_check, t_moist)
        )
        p_switch = interp1d(l_moist.m, p_check.m)(0)
        t_switch = interp1d(p_check.m, t_moist.m_as(units.kelvin))(p_switch)

        return p_switch.item()*units.mbar, t_switch.item()*units.kelvin

    def _nonentraining_properties(
            self, p_initial, t_initial, q_initial, l_initial, *, kind):
        """
        Calculate the properties of a non-entraining parcel.

        This is valid for both ascent and descent.

        Args:
            p_initial: Starting pressure of the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.
            kind: 'up' for updrafts, 'down' for downdrafts.

        Returns:
            Arrays of parcel temperatures, specific humidities and
            liquid ratios at the levels of interest. Any levels below
            (above) the initial level for updrafts (downdrafts) will
            be np.nan.
        """
        p_switch, t_switch = self._transition_point(
            p_initial, t_initial, q_initial, l_initial)

        t_final = np.full(self.pressure.size, np.nan)*units.kelvin
        q_final = np.full(self.pressure.size, np.nan)*units.dimensionless
        l_final = np.full(self.pressure.size, np.nan)*units.dimensionless

        # descent is moist adiabatic above the transition point and
        # dry adiabatic below the transition point. exclude any levels
        # below the initial level for updrafts, and any above the initial
        # level for downdrafts, to avoid unnecessary calculations.
        if kind == 'up':
            moist_levels = ((self.pressure <= p_switch)
                            & (self.pressure <= p_initial))
            dry_levels = ((self.pressure > p_switch)
                          & (self.pressure <= p_initial))
        else:
            moist_levels = ((self.pressure <= p_switch)
                            & (self.pressure >= p_initial))
            dry_levels = ((self.pressure > p_switch)
                          & (self.pressure >= p_initial))

        if np.any(moist_levels):
            t_final[moist_levels] = moist_lapse(
                self.pressure[moist_levels], t_switch,
                reference_pressure=p_switch, method='fast')
            q_final[moist_levels] = saturation_specific_humidity(
                self.pressure[moist_levels], t_final[moist_levels])
            l_final[moist_levels] = (
                q_initial + l_initial - q_final[moist_levels]
            )

        if np.any(dry_levels):
            t_final[dry_levels] = mpcalc.dry_lapse(
                self.pressure[dry_levels], t_switch,
                reference_pressure=p_switch)
            q_final[dry_levels] = saturation_specific_humidity(
                p_switch, t_switch)
            l_final[dry_levels] = 0*units.dimensionless

        return t_final, q_final, l_final

    def _updraft_properties(
            self, i_init, t_initial, q_initial, l_initial,
            entrainment_rate, l_crit):
        """
        Find the properties of a heterogeneous updraft parcel.

        The method follows Section 4 of Sherwood et al. (2013).

        Args:
            i_init: Index of the starting level for the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.
            entrainment_rate: Entrainment rate.
            l_crit: The critical liquid water content above which
                precipitation forms.

        Returns:
            Arrays of length pressure.size containing the average
            temperature, specific humidity, liquid content and buoyancy
            of the parcel, and the amount of liquid precipitated out,
            weighted by the mass mixing fraction of the components.
            Any levels below the initial level will be np.nan.

        References:
            SHERWOOD, SC, HERNANDEZ-DECKERS, D, COLIN, M & ROBINSON, F 2013,
            ‘Slippery Thermals and the Cumulus Entrainment Paradox’, Journal
            of the atmospheric sciences, vol. 70, no. 8, pp. 2426–2442.
        """
        # find the properties of the non-entraining initial 'core' component
        # of the parcel
        t_core, q_core, l_core = self._nonentraining_properties(
            self.pressure[i_init], t_initial, q_initial, l_initial, kind='up')
        r_core = mpcalc.mixing_ratio_from_specific_humidity(q_core)
        tv_core = mpcalc.virtual_temperature(t_core, r_core)
        b_core = (tv_core - self.t_virtual)/self.t_virtual*const.g

        # liquid content cannot exceed the critical value
        l_core_excess = np.maximum(l_core - l_crit, 0)
        l_disp_excess = np.maximum(self.l_disp - l_crit, 0)
        l_core_excess = np.where(np.isnan(l_core_excess), 0, l_core_excess)
        l_core -= l_core_excess
        l_disp = self.l_disp - l_disp_excess
        # currently l_core_excess[i] is the cumulative amount
        # precipitated out at or below level i. the true amount
        # precipitated out at level i is the increase in the
        # cumulative amount from level i+1 to level i:
        precip_core = np.pad(l_core_excess[:-1] - l_core_excess[1:], (0, 1))
        # similarly l_disp_excess[i,j] is the cumulative amount
        # precipitated out at or below level j by the parcel starting
        # at level i:
        precip_disp = l_disp_excess[:,:-1] - l_disp_excess[:,1:]
        precip_disp = np.pad(precip_disp, ((0, 0), (0, 1)))

        # following Eq. (8), (9) of Sherwood et al. (2013):
        # mixing_fraction[i,j] is the proportion of air originally in the
        # parcel at height[i] that remains in the parcel after
        # ascent/descent to height[j], i.e.,
        # mixing_fraction[i,j] = exp(-eps|height[i] - height[j]|).
        mixing_fraction = np.exp(-entrainment_rate*np.abs(
            np.atleast_2d(self.height) - np.atleast_2d(self.height).T
        ))

        # initialise unused array entries as nan
        t_mix = np.full(self.pressure.size, np.nan)*units.kelvin
        q_mix = np.full(self.pressure.size, np.nan)*units.dimensionless
        l_mix = np.full(self.pressure.size, np.nan)*units.dimensionless
        b_mix = np.full(self.pressure.size, np.nan)*units.meter/units.second**2
        precip_mix = np.zeros(self.pressure.size)*units.dimensionless
        # values at the initial height are the initial values, no entrainment
        t_mix[i_init] = t_core[i_init]
        q_mix[i_init] = q_core[i_init]
        l_mix[i_init] = l_core[i_init]
        b_mix[i_init] = b_core[i_init]
        precip_mix[i_init] = precip_core[i_init]

        # i_init - dir_ is the index of the first level the parcel
        # encounters after the initial level, and end is one index past
        # the last vertical level in the direction of travel (since range
        # and slice do not include the last value)
        for j in range(i_init-1, -1, -1):  # j is destination level
            if j == 0:
                i_range = slice(i_init, None, -1)
            else:
                i_range = slice(i_init, j-1, -1)
            z_interval = self.height[i_range].m_as(units.meter)

            # following Eq. (10) of Sherwood et al. (2013):
            # integrate the contributions of environmental parcels that
            # are entrained in between the initial and final levels using
            # Simpson's rule
            t_integrand = (
                self.t_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(units.kelvin/units.meter)
            # add a negative sign to the integral if z_interval is
            # decreasing (i.e., for a downdraft parcel) so that we are
            # always integrating upwards
            t_integral = simpson(t_integrand, z_interval)*units.kelvin
            t_mix[j] = t_core[j]*mixing_fraction[i_init,j] + t_integral

            # same procedure as above for the other variables
            q_integrand = (
                self.q_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.meter)
            q_integral = simpson(q_integrand, z_interval)
            q_integral *= units.dimensionless
            q_mix[j] = q_core[j]*mixing_fraction[i_init,j] + q_integral

            l_integrand = (
                l_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.meter)
            l_integral = simpson(l_integrand, z_interval)
            l_integral *= units.dimensionless
            l_mix[j] = l_core[j]*mixing_fraction[i_init,j] + l_integral

            b_integrand = (
                self.b_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.second**2)
            b_integral = simpson(b_integrand, z_interval)
            b_integral *= units.meter/units.second**2
            b_mix[j] = b_core[j]*mixing_fraction[i_init,j] + b_integral

            precip_integrand = (
                precip_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.meter)
            precip_integral = simpson(precip_integrand, z_interval)
            precip_integral *= units.dimensionless
            precip_mix[j] = (precip_core[j]*mixing_fraction[i_init,j]
                             + precip_integral)

        return t_mix, q_mix, l_mix, b_mix - l_mix*const.g, precip_mix

    def _downdraft_properties(
            self, i_init, t_initial, q_initial, l_initial, entrainment_rate):
        """
        Find the properties of a heterogeneous downdraft parcel.

        The method follows Section 4 of Sherwood et al. (2013).

        Args:
            i_init: Index of the starting level for the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.
            entrainment_rate: Entrainment rate.

        Returns:
            Arrays of length pressure.size containing the average
            temperature, specific humidity, liquid content and buoyancy
            of the parcel, weighted by the mass mixing fraction of the
            components. Any levels above the initial level will be np.nan.

        References:
            SHERWOOD, SC, HERNANDEZ-DECKERS, D, COLIN, M & ROBINSON, F 2013,
            ‘Slippery Thermals and the Cumulus Entrainment Paradox’, Journal
            of the atmospheric sciences, vol. 70, no. 8, pp. 2426–2442.
        """
        # find the properties of the non-entraining initial 'core' component
        # of the parcel
        t_core, q_core, l_core = self._nonentraining_properties(
            self.pressure[i_init], t_initial,
            q_initial, l_initial, kind='down')
        r_core = mpcalc.mixing_ratio_from_specific_humidity(q_core)
        tv_core = mpcalc.virtual_temperature(t_core, r_core)
        b_core = (tv_core - self.t_virtual)/self.t_virtual*const.g

        # following Eq. (8), (9) of Sherwood et al. (2013):
        # mixing_fraction[i,j] is the proportion of air originally in the
        # parcel at height[i] that remains in the parcel after
        # ascent/descent to height[j], i.e.,
        # mixing_fraction[i,j] = exp(-eps|height[i] - height[j]|).
        mixing_fraction = np.exp(-entrainment_rate*np.abs(
            np.atleast_2d(self.height) - np.atleast_2d(self.height).T
        ))

        # initialise unused array entries as nan
        t_mix = np.full(self.pressure.size, np.nan)*units.kelvin
        q_mix = np.full(self.pressure.size, np.nan)*units.dimensionless
        l_mix = np.full(self.pressure.size, np.nan)*units.dimensionless
        b_mix = np.full(self.pressure.size, np.nan)*units.meter/units.second**2
        # values at the initial height are the initial values, no entrainment
        t_mix[i_init] = t_core[i_init]
        q_mix[i_init] = q_core[i_init]
        l_mix[i_init] = l_core[i_init]
        b_mix[i_init] = b_core[i_init]

        # i_init - dir_ is the index of the first level the parcel
        # encounters after the initial level, and end is one index past
        # the last vertical level in the direction of travel (since range
        # and slice do not include the last value)
        for j in range(i_init+1, self.pressure.size):  # j is destination level
            z_interval = self.height[i_init:j+1].m_as(units.meter)

            # following Eq. (10) of Sherwood et al. (2013):
            # integrate the contributions of environmental parcels that
            # are entrained in between the initial and final levels using
            # Simpson's rule
            t_integrand = (
                self.t_disp[i_init:j+1,j].T*entrainment_rate
                * mixing_fraction[i_init:j+1,j].T
            ).m_as(units.kelvin/units.meter)
            # add a negative sign to the integral if z_interval is
            # decreasing (i.e., for a downdraft parcel) so that we are
            # always integrating upwards
            t_integral = -simpson(t_integrand, z_interval)*units.kelvin
            t_mix[j] = t_core[j]*mixing_fraction[i_init,j] + t_integral

            # same procedure as above for the other variables
            q_integrand = (
                self.q_disp[i_init:j+1,j].T*entrainment_rate
                * mixing_fraction[i_init:j+1,j].T
            ).m_as(1/units.meter)
            q_integral = -simpson(q_integrand, z_interval)
            q_integral *= units.dimensionless
            q_mix[j] = q_core[j]*mixing_fraction[i_init,j] + q_integral

            l_integrand = (
                self.l_disp[i_init:j+1,j].T*entrainment_rate
                * mixing_fraction[i_init:j+1,j].T
            ).m_as(1/units.meter)
            l_integral = -simpson(l_integrand, z_interval)
            l_integral *= units.dimensionless
            l_mix[j] = l_core[j]*mixing_fraction[i_init,j] + l_integral

            b_integrand = (
                self.b_disp[i_init:j+1,j].T*entrainment_rate
                * mixing_fraction[i_init:j+1,j].T
            ).m_as(1/units.second**2)
            b_integral = -simpson(b_integrand, z_interval)
            b_integral *= units.meter/units.second**2
            b_mix[j] = b_core[j]*mixing_fraction[i_init,j] + b_integral

        return t_mix, q_mix, l_mix, b_mix - l_mix*const.g

    def _velocity_profile(
            self, i_init, w_initial, buoyancy, drag=0/units.meter, *, kind):
        """
        Determine the downdraft velocity profile.

        Args:
            i_init: Index of the starting level.
            w_initial: Initial velocity. Should be non-positive for
                downdrafts and non-negative for updrafts.
            buoyancy: Array of parcel buoyancies at the given heights.
            drag: The drag coefficient.
            kind: 'up' for updrafts, 'down' for downdrafts.

        Returns:
            An array of length height.size containing the velocity profile.
            For updrafts (downdrafts), any entries for levels below (above)
            i_init or above (below) the highest (lowest) level reached will
            be np.nan.
        """
        velocity = np.full(self.height.size, np.nan)*units.meter/units.second
        velocity[i_init] = w_initial

        if kind == 'up':
            dir_ = 1
            end = -1
        else:
            dir_ = -1
            end = self.height.size
        # i_init - dir_ is the index of the first level the parcel
        # encounters after the initial level, and end is one index past
        # the last vertical level in the direction of travel (since range
        # and slice do not include the last value)
        for j in range(i_init - dir_, end, -dir_):
            if j - dir_ == -1:
                i_range = slice(i_init, None, -dir_)
            else:
                i_range = slice(i_init, j - dir_, -dir_)

            # implementation of the analytic solution using the integrating
            # factor method, approximating the integral using Simpson's
            # rule
            integrand = (np.exp(dir_*2*drag*self.height[i_range])
                         * buoyancy[i_range])
            integral = simpson(
                integrand.m_as(units.meter/units.second**2),
                self.height[i_range].m_as(units.meter)
            )*units.meter**2/units.second**2
            v_squared = (
                np.exp(dir_*2*drag*(self.height[i_init] - self.height[j]))
                * w_initial**2
                + 2*np.exp(-dir_*2*drag*self.height[j])*integral
            )

            if v_squared >= 0:
                velocity[j] = dir_*np.sqrt(v_squared)
            else:
                # v^2 < 0 indicates the thermal cannot reach that level or
                # any of the levels beyond: stop the calculation at this point
                velocity[j::-dir_] = np.nan
                break
        return velocity

    def _detrained_mass(
            self, velocity, buoyancy, dnu_db, entrainment_rate, *, kind):
        """
        Calculate the fractional mass detrained from a thermal at each level.

        Accounts for bulk detrainment and two-way mixing.

        Args:
            velocity: Vertical velocity of the thermal at each level,
                obtained from downdraft_velocity.
            buoyancy: Buoyancy of the thermal at each level.
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The dimensions of dnu_db
                should be time^2/length^2.
            entrainment_rate: Entrainment rate (should have dimensions
                of 1/length).
            kind: 'up' for updrafts, 'down' for downdrafts.

        Returns:
            Arrays of mass detrained and mass remaining at each level, as
            fractions of the original mass.
        """
        # velocity is nan for the levels the thermal does not reach:
        # use this to identify starting and ending levels
        reached_levels = np.argwhere(~np.isnan(velocity))
        if kind == 'up':
            # updraft: nu = 0 if b > 0, -b*dnu_db if b < 0
            nu = -np.minimum(buoyancy, 0*units.meter/units.second**2)*dnu_db
            i_init = np.max(reached_levels)
            i_end = np.min(reached_levels)
            dir_ = 1
        else:
            # downdraft: nu = 0 if b < 0, b*dnu_db if b < 0
            nu = np.maximum(buoyancy, 0*units.meter/units.second**2)*dnu_db
            i_init = np.min(reached_levels)
            i_end = np.max(reached_levels)
            dir_ = -1

        m_remaining = np.full(self.height.size, np.nan)*units.dimensionless
        m_remaining[i_init] = 1*units.dimensionless  # start with 100% mass
        for j in range(i_init - dir_, i_end - dir_, -dir_):
            if j - dir_ == -1:
                i_range = slice(i_init, None, -dir_)
            else:
                i_range = slice(i_init, j - dir_, -dir_)
            # fractional mass remaining = exp(integral_{z_0}^z nu(z') dz')
            integral = simpson(
                nu[i_range].m_as(1/units.meter),
                self.height[i_range].m_as(units.meter))
            m_remaining[j] = np.exp(-dir_*integral*units.dimensionless)

        # thickness of level j is approx. (z_j-1 - z_j+1)/2.
        # pad ends with zeroes to keep same length
        thickness = np.pad((self.height[:-2] - self.height[2:])/2, 1)
        thickness[0] = self.height[0] - self.height[1]  # topmost layer
        thickness[-1] = self.height[-2] - self.height[-1]  # surface layer
        # change in fractional mass is approx. m * nu * delta z
        m_deposited = m_remaining*(entrainment_rate + nu)*thickness
        return np.where(np.isnan(m_deposited), 0, m_deposited), m_remaining


class UpdraftResult:
    """Container for updraft calculation results."""

    def __init__(self):
        """Creates an instance of UpdraftResult."""
        self.temperature = None
        self.specific_humidity = None
        self.liquid_content = None
        self.precipitation = None
        self.buoyancy = None
        self.velocity = None
        self.m_detrained = None
        self.m_remaining = None
        self.t_detrained = None
        self.q_detrained = None
        self.l_detrained = None


class DowndraftResult:
    """Container for downdraft calculation results."""

    def __init__(self):
        """Creates an instance of DowndraftResult."""
        self.temperature = None
        self.specific_humidity = None
        self.liquid_content = None
        self.buoyancy = None
        self.velocity = None
        self.m_detrained = None
        self.m_remaining = None
        self.t_detrained = None
        self.q_detrained = None
        self.l_detrained = None


def equilibrate(pressure, t_initial, q_initial, l_initial):
    """
    Find parcel properties after phase equilibration.

    Args:
        pressure: Pressure during the change (constant).
        t_initial: Initial temperature of the parcel.
        q_initial: Initial specific humidity of the parcel.
        l_initial: Initial ratio of liquid mass to parcel mass.

    Returns:
        A tuple containing the final parcel temperature, specific
            humidity and liquid ratio.
    """
    q_sat_initial = saturation_specific_humidity(pressure, t_initial)
    if ((q_initial <= q_sat_initial and l_initial <= 0)
            or q_initial == q_sat_initial):
        # parcel is already in equilibrium
        return t_initial, q_initial, np.maximum(l_initial, 0)

    # to find the initial temperature after evaporation,first assume
    # that the parcel becomes saturated and therefore attains the
    # environmental wet bulb temperature
    theta_e = equivalent_potential_temperature(pressure, t_initial, q_initial)
    t_final = wetbulb(pressure, theta_e)
    q_final = saturation_specific_humidity(pressure, t_final)
    l_final = q_initial + l_initial - q_final

    # check if the assumption was realistic
    if l_final < 0:
        # if the liquid content resulting from evaporation to the point
        # of saturation is negative, this indicates that l_initial is
        # not large enough to saturate the parcel. We find the actual
        # resulting temperature using the conservation of equivalent
        # potential temperature during the evaporation process:
        # we use Newton's method to seek the temperature such that
        # the final equivelent potential temperature is unchanged.
        # As an initial guess, assume the temperature change is -L*dq/c_p
        t_final = t_initial - (const.water_heat_vaporization
                               * l_initial/const.dry_air_spec_heat_press)
        q_final = q_initial + l_initial
        l_final = 0*units.dimensionless
        for _ in range(3):
            value, slope = equivalent_potential_temperature(
                pressure, t_final, q_final, prime=True)
            t_final -= (value - theta_e)/slope

    return t_final, q_final, l_final
