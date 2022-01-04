"""Class for updraft and downdraft calculations."""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from metpy.units import units, concatenate
import metpy.calc as mpcalc
import metpy.constants as const

from dparcel.thermo import lcl_romps, moist_lapse, saturation_specific_humidity

class ThermalGenerator:
    """
    Collects functions for updrafts and downdrafts in a given sounding.

    Methods:

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
        b_disp = ((1 - l_disp)*tv_disp - tv_env)/tv_env*const.g

        self.pressure = pressure
        self.height = height
        self.temperature = temperature
        self.specific_humidity = specific_humidity
        self.t_virtual = tv_env
        self.t_disp = t_disp
        self.q_disp = q_disp
        self.l_disp = l_disp
        self.b_disp = b_disp

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
            liquid ratios at the levels of interest.
        """
        p_switch, t_switch = self._transition_point(
            p_initial, t_initial, q_initial, l_initial)

        t_final = np.zeros(self.pressure.size)*units.kelvin
        q_final = np.zeros(self.pressure.size)*units.dimensionless
        l_final = np.zeros(self.pressure.size)*units.dimensionless

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

    def _heterogeneous_properties(
            self, i_init, t_initial, q_initial, l_initial,
            entrainment_rate, *, kind):
        """
        Find the properties of a heterogeneous parcel.

        The method follows Section 4 of Sherwood et al. (2013).

        Args:
            i_init: Index of the starting level for the parcel.
            t_initial: Initial temperature of the parcel.
            q_initial: Initial specific humidity of the parcel.
            l_initial: Initial liquid water ratio of the parcel.
            entrainment_rate: Entrainment rate.
            kind: 'up' for updrafts, 'down' for downdrafts.

        Returns:
            Arrays of length pressure.size containing the average
            temperature, specific humidity, liquid content and buoyancy
            of the parcel, weighted by the mass mixing fraction of the
            components.

        References:
            SHERWOOD, SC, HERNANDEZ-DECKERS, D, COLIN, M & ROBINSON, F 2013,
            ‘Slippery Thermals and the Cumulus Entrainment Paradox’, Journal
            of the atmospheric sciences, vol. 70, no. 8, pp. 2426–2442.
        """
        # find the properties of the non-entraining initial 'core' component
        # of the parcel
        t_core, q_core, l_core = self._nonentraining_properties(
            self.pressure[i_init], t_initial, q_initial, l_initial, kind=kind)
        r_core = mpcalc.mixing_ratio_from_specific_humidity(q_core)
        tv_core = mpcalc.virtual_temperature(t_core, r_core)
        b_core = ((1 - l_core)*tv_core - self.t_virtual)/self.t_virtual*const.g

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

        if kind == 'up':
            dir_ = 1
            end = -1
        else:
            dir_ = -1
            end = self.pressure.size
        # i_init - dir_ is the index of the first level the parcel
        # encounters after the initial level, and end is one index past
        # the last vertical level in the direction of travel (since range
        # and slice do not include the last value)
        for j in range(i_init - dir_, end, -dir_):  # j is destination level
            if j - dir_ == -1:
                i_range = slice(i_init, None, -dir_)
            else:
                i_range = slice(i_init, j - dir_, -dir_)
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
            t_integral = dir_*simpson(t_integrand, z_interval)*units.kelvin
            t_mix[j] = t_core[j]*mixing_fraction[i_init,j] + t_integral

            # same procedure as above for the other variables
            q_integrand = (
                self.q_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.meter)
            q_integral = dir_*simpson(q_integrand, z_interval)
            q_integral *= units.dimensionless
            q_mix[j] = q_core[j]*mixing_fraction[i_init,j] + q_integral

            l_integrand = (
                self.l_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.meter)
            l_integral = dir_*simpson(l_integrand, z_interval)
            l_integral *= units.dimensionless
            l_mix[j] = l_core[j]*mixing_fraction[i_init,j] + l_integral

            b_integrand = (
                self.b_disp[i_range,j].T*entrainment_rate
                * mixing_fraction[i_range,j].T
            ).m_as(1/units.second**2)
            b_integral = dir_*simpson(b_integrand, z_interval)
            b_integral *= units.meter/units.second**2
            b_mix[j] = b_core[j]*mixing_fraction[i_init,j] + b_integral

        return t_mix, q_mix, l_mix, b_mix

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

            # v^2 < 0 indicates the thermal cannot reach that level
            velocity[j] = dir_*np.sqrt(v_squared) if v_squared >= 0 else np.nan
        return velocity

    def _detrained_mass(self, velocity, buoyancy, dnu_db, *, kind):
        """
        Calculate the fractional mass detrained from a thermal at each level.

        Args:
            velocity: Vertical velocity of the thermal at each level,
                obtained from downdraft_velocity.
            buoyancy: Buoyancy of the thermal at each level.
            dnu_db: The proportionality constant defining the detrainent
                rate nu. When the buoyancy b is negative, nu is zero,
                and when b > 0, nu = b*dnu_db. The units of dnu_db should be
                s^2/m^2.
            kind: 'up' for updrafts, 'down' for downdrafts.

        Returns:
            An array of mass detrained at each level, as a fraction of the
            original mass.
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

        m_remaining = np.zeros(self.height.size)*units.dimensionless
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
        m_deposited = m_remaining*nu*thickness
        return np.where(np.isnan(m_deposited), 0*units.dimensionless,
                        m_deposited)
