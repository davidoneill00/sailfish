"""
Code to solve the Kepler two-body problem, and its inverse.
"""

from typing import NamedTuple
from math import sin, cos, sqrt, atan2, pi, floor

"""
Newton's gravitational constant is G=1.0, so mass M really means G M.
"""
NEWTON_G = 1.0


class PointMass(NamedTuple):
    """
    The mass, 2D position, and 2D velocity of a point-like particle
    """

    mass: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float

    @property
    def kinetic_energy(self) -> float:
        """
        The kinetic energy of a point mass
        """
        vx = self.velocity_x
        vy = self.velocity_y
        return 0.5 * self.mass * (vx * vx + vy * vy)

    @property
    def angular_momentum(self) -> float:
        """
        The angular momentum of a point mass
        """
        x = self.position_x
        y = self.position_y
        vx = self.velocity_x
        vy = self.velocity_y
        return self.mass * (x * vy - y * vx)

    def gravitational_potential(
        self, x: float, y: float, softening_length: float
    ) -> float:
        """
        Return the gravitational potential of a point mass, with softening.
        """
        dx = x - self.position_x
        dy = y - self.position_y
        r2 = dx * dx + dy * dy
        s2 = softening_length.powi(2)
        return -NEWTON_G * self.mass / sqrt(r2 + s2)

    def gravitational_acceleration(
        p, x: float, y: float, softening_length: float
    ) -> (float, float):
        """
        Return the gravitational acceleration due to a point mass.
        """
        dx = x - self.position_x
        dy = y - self.position_y
        r2 = dx * dx + dy * dy
        s2 = softening_length ** 2.0
        ax = -NEWTON_G * self.mass / (r2 + s2) ** 1.5 * dx
        ay = -NEWTON_G * self.mass / (r2 + s2) ** 1.5 * dy
        return (ax, ay)

    def perturb(
        self, dm: float = 0.0, dpx: float = 0.0, dpy: float = 0.0
    ) -> "PointMass":
        """
        Perturb the mass and momentum of a point mass.

        Since the point mass maintains a velocity rather than momentum,
        the velocity is changed according to

        dv = (dp - v dm) / m
        """
        return self._replace(
            mass=self.mass + dm,
            velocity_x=velocity_x + (dpx - self.velocity_x * dm) / self.mass,
            velocity_y=velocity_y + (dpy - self.velocity_y * dm) / self.mass,
        )


class OrbitalState(NamedTuple):
    primary: PointMass
    secondary: PointMass

    @property
    def total_mass(self) -> float:
        """
        The sum of the two point masses
        """
        return self[0].mass + self[1].mass

    @property
    def mass_ratio(self) -> float:
        """
        The system mass ratio, secondary / primary
        """
        return self[1].mass / self[0].mass

    @property
    def reduced_mass(self):
        """
        The system reduced mass, M1 * M2 / (M1 + M2)
        """
        return self[0].mass * self[1].mass / self.total_mass

    @property
    def separation(self) -> float:
        """
        The orbital separation

        This will always be the semi-major axis if the eccentricity is zero.
        """
        x1 = self[0].position_x
        y1 = self[0].position_y
        x2 = self[1].position_x
        y2 = self[1].position_y
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @property
    def semimajor_axis(self) -> float:
        try:
            return  -0.5 * NEWTON_G * self[0].mass * self[1].mass / self.total_energy
        except:
            return 0.

    @property
    def eccentricity(self):
        """
        The system eccentricity, secondary / primary
        """
        M   = self.total_mass
        mu  = self.reduced_mass
        a   = self.semimajor_axis
        L   = self.angular_momentum
        try:
            ecc = sqrt(1 - clamp_between_zero_and_one(L**2 / M / mu**2 / a))
            return ecc
        except:
            return 0.
    
    @property
    def total_energy(self) -> float:
        """
        The system total energy
        """
        return self.kinetic_energy - NEWTON_G * self[0].mass * self[1].mass / self.separation

    @property
    def kinetic_energy(self) -> float:
        """
        The total kinetic energy of the system
        """
        return self[0].kinetic_energy + self[1].kinetic_energy

    @property
    def angular_momentum(self) -> float:
        """
        The total anuglar momentum of the system
        """
        return self[0].angular_momentum + self[1].angular_momentum

    def gravitational_potential(
        self, x: float, y: float, softening_length: float
    ) -> float:
        """
        Return the combined gravitational potential at a point, with softening.
        """
        p0 = self[0].gravitational_potential(x, y, softening_length)
        p1 = self[1].gravitational_potential(x, y, softening_length)
        return p0 + p1

    def gravitational_acceleration(
        self, x: float, y: float, softening_length: float
    ) -> float:
        """
        Return the combined gravitational acceleration at a point, with softening.
        """
        a0 = self[0].gravitational_acceleration(x, y, softening_length)
        a1 = self[1].gravitational_acceleration(x, y, softening_length)
        return (a0[0] + a1[0], a0[1] + a1[1])

    def perturb(
        self, dm1: float, dm2: float, dpx1: float, dpx2: float, dpy1: float, dpy2: float
    ) -> "OrbitalState":
        """
        Returns a new orbital state vector if this one is perturbed by the
        given masses and momenta.

        - :code:`dm1`   Mass added to the primary
        - :code:`dm2`   Mass added to the secondary
        - :code:`dpx1`  Impulse (x) added to the primary
        - :code:`dpx2`  Impulse (x) added to the secondary
        - :code:`dpy1`  Impulse (y) added to the primary
        - :code:`dpy2`  Impulse (y) added to the secondary
        """

        return OrbitalState(
            self[0].perturb_mass_and_momentum(dm1, dpx1, dpy1),
            self[1].perturb_mass_and_momentum(dm2, dpx2, dpy2),
        )

    def true_anomaly(self, t: float) -> float:
        from numpy import arctan2
        """
        This function determines the true anomaly from the
        orbital state vector and an absolute time.
        """
        c1 = self[0]
        c2 = self[1]

        # component masses, total mass, and mass ratio
        m1 = c1.mass
        m2 = c2.mass
        m = m1 + m2
        q = m2 / m1

        # position and velocity of the CM frame
        x_cm = (c1.position_x * c1.mass + c2.position_x * c2.mass) / m
        y_cm = (c1.position_y * c1.mass + c2.position_y * c2.mass) / m
        vx_cm = (c1.velocity_x * c1.mass + c2.velocity_x * c2.mass) / m
        vy_cm = (c1.velocity_y * c1.mass + c2.velocity_y * c2.mass) / m

        # positions and velocities of the components in the CM frame
        x1 = c1.position_x - x_cm
        y1 = c1.position_y - y_cm
        x2 = c2.position_x - x_cm
        y2 = c2.position_y - y_cm

        return arctan2(y1,x1)

    def orbital_parameters(self, t: float) -> ("OrbitalElements", "OrbitalOrientation"):
        """
        Compute the inverse Kepler two-body problem.

        This function determines the orbital elements and orientation from the
        orbital state vector and an absolute time.
        """
        c1 = self[0]
        c2 = self[1]

        # component masses, total mass, and mass ratio
        m1 = c1.mass
        m2 = c2.mass
        m = m1 + m2
        q = m2 / m1

        # position and velocity of the CM frame
        x_cm = (c1.position_x * c1.mass + c2.position_x * c2.mass) / m
        y_cm = (c1.position_y * c1.mass + c2.position_y * c2.mass) / m
        vx_cm = (c1.velocity_x * c1.mass + c2.velocity_x * c2.mass) / m
        vy_cm = (c1.velocity_y * c1.mass + c2.velocity_y * c2.mass) / m

        # positions and velocities of the components in the CM frame
        x1 = c1.position_x - x_cm
        y1 = c1.position_y - y_cm
        x2 = c2.position_x - x_cm
        y2 = c2.position_y - y_cm
        r1 = sqrt(x1 * x1 + y1 * y1)
        r2 = sqrt(x2 * x2 + y2 * y2)
        vx1 = c1.velocity_x - vx_cm
        vy1 = c1.velocity_y - vy_cm
        vx2 = c2.velocity_x - vx_cm
        vy2 = c2.velocity_y - vy_cm
        vf1 = -vx1 * y1 / r1 + vy1 * x1 / r1
        vf2 = -vx2 * y2 / r2 + vy2 * x2 / r2
        v1 = sqrt(vx1 * vx1 + vy1 * vy1)

        # energy and angular momentum (t := kinetic energy, l := angular
        # momentum, h := total energy)
        t1 = 0.5 * m1 * (vx1 * vx1 + vy1 * vy1)
        t2 = 0.5 * m2 * (vx2 * vx2 + vy2 * vy2)
        l1 = m1 * r1 * vf1
        l2 = m2 * r2 * vf2
        r = r1 + r2
        l = l1 + l2
        h = t1 + t2 - NEWTON_G * m1 * m2 / r

        if h >= 0.0:
            raise ValueError("the orbit is unbound")

        # semi-major, semi-minor axes eccentricity, apsides
        a = -0.5 * NEWTON_G * m1 * m2 / h
        b = sqrt(-0.5 * l * l / h * (m1 + m2) / (m1 * m2))
        e = sqrt(clamp_between_zero_and_one(1.0 - b * b / a / a))
        omega = sqrt(NEWTON_G * m / a / a / a)

        # semi-major and semi-minor axes of the primary
        a1 = a * q / (1.0 + q)
        b1 = b * q / (1.0 + q)

        # cos of nu and f: phase angle and true anomaly
        if e == 0.0:
            cn = x1 / r1
        else:
            cn = (1.0 - r1 / a1) / e
        cf = a1 / r1 * (cn - e)

        # sin of nu and f
        if e == 0.0:
            sn = y1 / r1
        else:
            sn = (vx1 * x1 + vy1 * y1) / (e * v1 * r1) * sqrt(1.0 - e * e * cn * cn)

        sf = (b1 / r1) * sn

        # cos and sin of eccentric anomaly
        ck = (e + cf) / (1.0 + e * cf)
        sk = sqrt(1.0 - e * e) * sf / (1.0 + e * cf)

        # mean anomaly and tau
        k = atan2(sk, ck)
        n = k - e * sk
        tau = t - n / omega

        # cartesian components of semi-major axis, and the argument of periapse
        ax = (cn - e) * x1 + sn * sqrt(1.0 - e * e) * y1
        ay = (cn - e) * y1 - sn * sqrt(1.0 - e * e) * x1
        if e>1e-4:
            pomega = atan2(ay, ax)
        else:
            pomega = 0

        # final result
        elements = OrbitalElements(a, m, q, e)
        orientation = OrbitalOrientation(x_cm, y_cm, vx_cm, vy_cm, pomega, tau)

        return elements, orientation


class OrbitalOrientation(NamedTuple):
    """
    The position, velocity, and orientation of a two-body orbit
    """

    cm_position_x: float
    cm_position_y: float
    cm_velocity_x: float
    cm_velocity_y: float
    periapse_argument: float
    periapse_time: float


class OrbitalElements(NamedTuple):
    """
    The orbital elements of a two-body system on a bound orbit
    """

    semimajor_axis: float
    total_mass: float
    mass_ratio: float
    eccentricity: float


    @property
    def omega(self) -> float:
        """
        The orbital angular frequency
        """
        m = self.total_mass
        a = self.semimajor_axis
        return sqrt(NEWTON_G * m / a / a / a)

    @property
    def period(self) -> float:
        """
        The orbital period
        """
        return 2.0 * pi / self.omega

    @property
    def angular_momentum(self) -> float:
        """
        The orbital angular momentum
        """
        a = self.semimajor_axis
        m = self.total_mass
        q = self.mass_ratio
        e = self.eccentricity
        m1 = m / (1.0 + q)
        m2 = m - m1
        return m1 * m2 / m * sqrt(NEWTON_G * m * a * (1.0 - e * e))
    '''
    def orbital_state_from_eccentric_anomaly(
        self, eccentric_anomaly: float
    ) -> OrbitalState:
        """
        Compute the orbital state, given the eccentric anomaly.
        """
        a = self.semimajor_axis
        m = self.total_mass
        q = self.mass_ratio
        e = self.eccentricity
        w = self.omega
        m1 = m / (1.0 + q)
        m2 = m - m1
        ck = cos(eccentric_anomaly)
        sk = sin(eccentric_anomaly)
        x1 = -a * q / (1.0 + q) * (e - ck)
        y1 = +a * q / (1.0 + q) * (sk) * sqrt(1.0 - e * e)
        x2 = -x1 / q
        y2 = -y1 / q
        vx1 = -a * q / (1.0 + q) * w / (1.0 - e * ck) * sk
        vy1 = +a * q / (1.0 + q) * w / (1.0 - e * ck) * ck * sqrt(1.0 - e * e)
        vx2 = -vx1 / q
        vy2 = -vy1 / q
        c1 = PointMass(m1, x1, y1, vx1, vy1)
        c2 = PointMass(m2, x2, y2, vx2, vy2)
        return OrbitalState(c1, c2)

    def eccentric_anomaly(self, time_since_periapse: float) -> float:
        """
        Compute the eccentric anomaly from the time since any periapse.
        """
        p = self.period
        t = time_since_periapse - self.period * floor(time_since_periapse / p)
        e = self.eccentricity
        n = self.omega * t  # n := mean anomaly M
        f = lambda k: k - e * sin(k) - n  # k := eccentric anomaly E
        g = lambda k: 1.0 - e * cos(k)
        return solve_newton_rapheson(f, g, n)

    def orbital_state(self, time_since_periapse: float) -> OrbitalState:
        """
        Compute the orbital state vector from the time since any periapse.
        """
        E = self.eccentric_anomaly(time_since_periapse)
        return self.orbital_state_from_eccentric_anomaly(E)

    def orbital_state_with_orientation(
        self, absolute_time, orientation: OrbitalOrientation
    ) -> OrbitalState:
        """
        Compute the orbital state from an absolute time and orientation.
        """
        t = absolute_time - orientation.periapse_time
        E = self.eccentric_anomaly(t)
        state = self.orbital_state_from_eccentric_anomaly(E)

        m1 = state[0].mass
        m2 = state[1].mass
        x1 = state[0].position_x
        x2 = state[1].position_x
        y1 = state[0].position_y
        y2 = state[1].position_y
        vx1 = state[0].velocity_x
        vx2 = state[1].velocity_x
        vy1 = state[0].velocity_y
        vy2 = state[1].velocity_y

        c = cos(-orientation.periapse_argument)
        s = sin(-orientation.periapse_argument)

        x1p = +x1 * c + y1 * s + orientation.cm_position_x
        y1p = -x1 * s + y1 * c + orientation.cm_position_y
        x2p = +x2 * c + y2 * s + orientation.cm_position_x
        y2p = -x2 * s + y2 * c + orientation.cm_position_y
        vx1p = +vx1 * c + vy1 * s + orientation.cm_velocity_x
        vy1p = -vx1 * s + vy1 * c + orientation.cm_velocity_y
        vx2p = +vx2 * c + vy2 * s + orientation.cm_velocity_x
        vy2p = -vx2 * s + vy2 * c + orientation.cm_velocity_y

        c1 = PointMass(m1, x1p, y1p, vx1p, vy1p)
        c2 = PointMass(m2, x2p, y2p, vx2p, vy2p)

        return OrbitalState(c1, c2)
    '''

def solve_newton_rapheson(f, g, x: float) -> float:
    n = 0
    while abs(f(x)) > 1e-15:
        x -= f(x) / g(x)
        n += 1
        if n > 10:
            raise ValueError("solve_newton_rapheson: no solution")
    return x


def clamp_between_zero_and_one(x: float) -> float:
    return min(1.0, max(0.0, x))
