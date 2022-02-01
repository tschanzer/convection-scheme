"""Class for updraft and downdraft calculations."""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from metpy.units import units, concatenate
import metpy.calc as mpcalc
import metpy.constants as const

from pint.errors import OffsetUnitCalculusError

from dparcel.thermo import (lcl_romps, moist_lapse_dj, wetbulb,
                            equivalent_potential_temperature,
                            saturation_specific_humidity)


class StochasticThermalGenerator:
    """
    Calculates coupled updrafts and downdrafts using a stochastic model.

    The stochastic model follows Romps and Kuang (2010).

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

    References:
        Romps, D. M., & Kuang, Z. (2010). Nature versus nurture in
        shallow convection. Journal of the Atmospheric Sciences, 67(5),
        1655–1666. https://doi.org/10.1175/2009JAS3307.1
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
                t_disp[i,:lcl_index[i]] = moist_lapse_dj(
                    pressure[:lcl_index[i]], t_lcl[i],
                    reference_pressure=p_lcl[i])
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
        self.rng = np.random.default_rng(seed=0)

    def updraft(
            self, i_init, t_perturb, q_perturb, l_initial, w_initial,
            lambda_, sigma, drag, l_crit, basic=False):
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
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            l_crit: The critical liquid water content above which
                precipitation forms.
            basic: Set to True to skip calculation of the detrained
                air properties.

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
                l_initial, lambda_, sigma, l_crit)
        )

        velocity = self._velocity_profile(
            i_init, w_initial, buoyancy, drag, kind='up')
        # ensure no precipitation above max height
        precipitation = np.where(np.isnan(velocity), 0, precipitation)

        if not basic:
            t_detrained = np.full(self.pressure.size, np.nan)
            t_detrained *= units.kelvin
            q_detrained = np.full(self.pressure.size, np.nan)
            q_detrained *= units.dimensionless
            l_detrained = np.full(self.pressure.size, np.nan)
            l_detrained *= units.dimensionless
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
        if not basic:
            result.t_detrained = t_detrained
            result.q_detrained = q_detrained
            result.l_detrained = l_detrained
        return result

    def downdraft(
            self, i_init, t_initial, q_initial, l_initial, w_initial,
            lambda_, sigma, drag, basic=False):
        """
        Calculate the properties associated with a descending thermal.

        Args:
            i_init: Index of the initiation level.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity
            l_initial: Initial liquid water content (mass of liquid
                per unit total mass).
            w_initial: Initial velocity (must be non-negative).
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            basic: Set to True to skip calculation of the detrained
                air properties.

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
                - **t_detrained** -- Array containing the temperature of
                  detrained air at each level.
                - **q_detrained** -- Array containing the specific
                  humidity of detrained air at each level.
                - **l_detrained** -- Array containing the liquid
                  content of detrained air at each level.
        """
        temperature, specific_humidity, liquid_content, buoyancy = (
            self._downdraft_properties(
                i_init, t_initial, q_initial, l_initial, lambda_, sigma)
        )
        velocity = self._velocity_profile(
            i_init, w_initial, buoyancy, drag, kind='down')

        if not basic:
            t_detrained = np.full(self.pressure.size, np.nan)
            t_detrained *= units.kelvin
            q_detrained = np.full(self.pressure.size, np.nan)
            q_detrained *= units.dimensionless
            l_detrained = np.full(self.pressure.size, np.nan)
            l_detrained *= units.dimensionless
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
        if not basic:
            result.t_detrained = t_detrained
            result.q_detrained = q_detrained
            result.l_detrained = l_detrained
        return result

    def precipitation_downdraft(
            self, i_init, delta_Q, w_initial,
            lambda_, sigma, drag, basic=False):
        """
        Calculate the properties associated with a descending thermal.

        The descent is assumed to be triggered by the evaporation of
        precipitation into an environmental parcel.

        Args:
            i_init: Index of the initiation level.
            delta_Q: Total mass of liquid water initially introduced
                into the environmental parcel at i_init.
            w_initial: Initial velocity (must be non-negative).
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            drag: Drag coefficient for determining parcel velocity.
                Should have dimensions of 1/length.
            basic: Set to True to skip calculation of the detrained
                air properties.

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

        return self.downdraft(
            i_init, t_initial, q_initial, l_initial, w_initial,
            lambda_, sigma, drag, basic=basic)

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
        t_moist = moist_lapse_dj(
            p_check, t_initial, reference_pressure=p_initial)
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
        t_moist = moist_lapse_dj(
            p_check, t_initial, reference_pressure=p_initial)
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
            t_final[moist_levels] = moist_lapse_dj(
                self.pressure[moist_levels], t_switch,
                reference_pressure=p_switch)
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
            lambda_, sigma, l_crit):
        """
        Find the properties of a heterogeneous updraft parcel.

        Args:
            i_init: Index of the starting level for the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.
            l_crit: The critical liquid water content above which
                precipitation forms.

        Returns:
            Arrays of length pressure.size containing the average
            temperature, specific humidity, liquid content and buoyancy
            of the parcel, and the amount of liquid precipitated out,
            weighted by the mass mixing fraction of the components.
            Any levels below the initial level will be np.nan.
        """
        # generate n exponentially distributed distances between
        # successive entrainment events, where n is twice the expected
        # number of entrainment events between the initial level and
        # the top of the sounding
        n_samples = 2*np.ceil((self.height[0] - self.height[i_init])/lambda_)
        intervals = self.rng.exponential(
            scale=lambda_.m_as(units.meter), size=int(n_samples))

        # take the column vector of distances between events,
        # stack copies side by side to form a square matrix,
        # set entries below the diagonal to zero then sum along the
        # vertical axis to get a vector of distances between the initial
        # level and each entrainment event. then add on the initial
        # height to get the locations of the events.
        intervals = np.tile(intervals, (intervals.size, 1)).T
        z_entrain = (self.height[i_init]
                     + np.sum(np.triu(intervals), axis=0)*units.meter)

        # exclude events above the top of the sounding
        z_entrain = z_entrain[z_entrain < self.height[0]]
        # for each event's location, find the index of the closest
        # sounding level
        i_entrain = np.argmin(np.abs(
            np.atleast_2d(self.height).T - np.atleast_2d(z_entrain)
        ), axis=0)
        # exclude any repetitions (i.e., max one event per level)
        i_entrain = np.flip(np.unique(i_entrain))
        # get a boolean array, true if level i has an event, else false
        entrainment_occurs = np.isin(np.arange(self.height.size), i_entrain)

        # generate exponentially distributed mass fractions to be
        # entrained at each event
        m_entrain = self.rng.exponential(
            scale=sigma.m_as(units.dimensionless), size=len(i_entrain)
        )*units.dimensionless

        # find the properties of the original component
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

        # stack the row vectors containing the properties of each
        # entrained component into matrices
        t_all = np.vstack([self.t_disp[entrainment_occurs,:], t_core])
        q_all = np.vstack([self.q_disp[entrainment_occurs,:], q_core])
        l_all = np.vstack([l_disp[entrainment_occurs,:], l_core])
        b_all = np.vstack([self.b_disp[entrainment_occurs,:], b_core])
        precip_all = np.vstack(
            [precip_disp[entrainment_occurs,:], precip_core])

        # m_all[i,j] will be the mass fraction of component number i
        # at level j. initially only one component is present:
        m_all = np.ones((1, self.height.size))*units.dimensionless
        m_all[0,i_init+1:] = np.nan

        # at each event, we introduce a mass fraction m, and all the
        # mass fractions present are scaled by a factor of 1/(1 + m).
        for i, m in zip(i_entrain, m_entrain):
            m_new = np.ones(self.height.size)*m
            m_new[i+1:] = 0
            m_all = np.insert(m_all, 0, m_new, axis=0)
            m_all[:,:i+1] *= 1/(1 + m)

        # the final properties are the weighted average component
        # properties weighted by mass fraction:
        t_final = np.sum(t_all*m_all, axis=0)
        q_final = np.sum(q_all*m_all, axis=0)
        l_final = np.sum(l_all*m_all, axis=0)
        b_final = np.sum(b_all*m_all, axis=0)
        precip_final = np.sum(precip_all*m_all, axis=0)

        return (t_final, q_final, l_final,
                b_final - l_final*const.g, precip_final)

    def _downdraft_properties(
            self, i_init, t_initial, q_initial,
            l_initial, lambda_, sigma):
        """
        Find the properties of a heterogeneous downdraft parcel.

        Args:
            i_init: Index of the starting level for the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.
            lambda_: Mean distance between entrainment events.
            sigma: Mean fractional mass entrained in events.

        Returns:
            Arrays of length pressure.size containing the average
            temperature, specific humidity, liquid content and buoyancy
            of the parcel, weighted by the mass mixing fraction of the
            components. Any levels above the initial level will be np.nan.
        """
        # generate n exponentially distributed distances between
        # successive entrainment events, where n is twice the expected
        # number of entrainment events between the initial level and
        # the bottom of the sounding
        n_samples = 2*np.ceil((self.height[i_init] - self.height[-1])/lambda_)
        intervals = self.rng.exponential(
            scale=lambda_.m_as(units.meter), size=int(n_samples))

        # take the column vector of distances between events,
        # stack copies side by side to form a square matrix,
        # set entries below the diagonal to zero then sum along the
        # vertical axis to get a vector of distances between the initial
        # level and each entrainment event. then subtract these from
        # the initial height to get the locations of the events.
        intervals = np.tile(intervals, (intervals.size, 1)).T
        z_entrain = (self.height[i_init]
                     - np.sum(np.triu(intervals), axis=0)*units.meter)

        # exclude events below the bottom of the sounding
        z_entrain = z_entrain[z_entrain > self.height[-1]]
        # for each event's location, find the index of the closest
        # sounding level
        i_entrain = np.argmin(np.abs(
            np.atleast_2d(self.height).T - np.atleast_2d(z_entrain)
        ), axis=0)
        # exclude any repetitions (i.e., max one event per level)
        i_entrain = np.unique(i_entrain)
        # get a boolean array, true if level i has an event, else false
        entrainment_occurs = np.isin(np.arange(self.height.size), i_entrain)

        # generate exponentially distributed mass fractions to be
        # entrained at each event
        m_entrain = self.rng.exponential(
            scale=sigma.m_as(units.dimensionless), size=len(i_entrain)
        )*units.dimensionless

        # find the properties of the original component
        t_core, q_core, l_core = self._nonentraining_properties(
            self.pressure[i_init], t_initial,
            q_initial, l_initial, kind='down')
        r_core = mpcalc.mixing_ratio_from_specific_humidity(q_core)
        tv_core = mpcalc.virtual_temperature(t_core, r_core)
        b_core = (tv_core - self.t_virtual)/self.t_virtual*const.g

        # stack the row vectors containing the properties of each
        # entrained component into matrices
        t_all = np.vstack([t_core, self.t_disp[entrainment_occurs,:]])
        q_all = np.vstack([q_core, self.q_disp[entrainment_occurs,:]])
        l_all = np.vstack([l_core, self.l_disp[entrainment_occurs,:]])
        b_all = np.vstack([b_core, self.b_disp[entrainment_occurs,:]])

        # m_all[i,j] will be the mass fraction of component number i
        # at level j. initially only one component is present:
        m_all = np.ones((1, self.height.size))*units.dimensionless
        m_all[0,:i_init] = np.nan

        # at each event, we introduce a mass fraction m, and all the
        # mass fractions present are scaled by a factor of 1/(1 + m).
        for i, m in zip(i_entrain, m_entrain):
            m_new = np.ones(self.height.size)*m
            m_new[:i] = 0
            m_all = np.insert(m_all, m_all.shape[0], m_new, axis=0)
            m_all[:,i:] *= 1/(1 + m)

        # the final properties are the weighted average component
        # properties weighted by mass fraction:
        t_final = np.sum(t_all*m_all, axis=0)
        q_final = np.sum(q_all*m_all, axis=0)
        l_final = np.sum(l_all*m_all, axis=0)
        b_final = np.sum(b_all*m_all, axis=0)

        return t_final, q_final, l_final, b_final - l_final*const.g

    def _velocity_profile(
            self, i_init, w_initial, buoyancy, drag=0/units.meter, *, kind):
        """
        Determine the up/downdraft velocity profile.

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
                break
        return velocity


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
    # that the parcel becomes saturated and therefore attains its
    # wet bulb temperature
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
        # the final equivalent potential temperature is unchanged.
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
