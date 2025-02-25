#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace dacti::geom {

	struct Polygon {
		Eigen::MatrixXd vertices;
	};

	/**
     * @brief Base class for representing a curve.
     */
	struct BaseCurve {
		/**
         * @brief Evaluate the curve at parameter a
         * @return A 2D point on the curve
         */
		virtual Eigen::Vector2d operator()(scalar_t a) const = 0;
		/** 
         * @brief Check if the curve is closed
         */
		virtual bool is_closed() const = 0;
	};


	struct Ellipse : public BaseCurve {
		scalar_t a;    // Semi-major axis
		scalar_t b;    // Semi-minor axis

		Ellipse(scalar_t a, scalar_t b) :
		    a(a), b(b) {}

		Eigen::Vector2d operator()(scalar_t t) const override {
			return Eigen::Vector2d(a * std::cos(t), b * std::sin(t));
		}
		bool is_closed() const override { return true; }
	};


	struct Circle : public BaseCurve {
		scalar_t r;    // Radius of the circle

		Circle(scalar_t r) :
		    r(r) {}

		Eigen::Vector2d operator()(scalar_t t) const override {
			return Eigen::Vector2d(r * std::cos(t), r * std::sin(t));
		}
		bool is_closed() const override { return true; }
	};


	struct NACA4Airfoil : public BaseCurve {
		scalar_t m;    // Maximum camber
		scalar_t p;    // Position of maximum camber
		scalar_t t;    // Maximum thickness

		NACA4Airfoil(scalar_t m, scalar_t p, scalar_t t) :
		    m(m), p(p), t(t) {}

		Eigen::Vector2d operator()(scalar_t x) const override {

			scalar_t sgn = (x >= 0) ? 1.0 : -1.0;

			x *= sgn;

			auto yt = [this](scalar_t x) {
				return 5 * t * (0.2969 * std::sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1036 * x * x * x * x);
			};

			auto yc = [this](scalar_t x) {
				if (x < p) {
					return m * (2 * p * x - x * x) / (p * p);
				} else {
					return m * (1 - 2 * p + 2 * p * x - x * x) / ((1 - p) * (1 - p));
				}
			};

			auto dyc_dx = [this](scalar_t x) {
				if (x < p) {
					return 2 * m * (p - x) / (p * p);
				} else {
					return 2 * m * (p - x) / ((1 - p) * (1 - p));
				}
			};

			scalar_t theta = std::atan(dyc_dx(x));
			scalar_t y_c   = yc(x);
			scalar_t y_t   = yt(x);

			return Eigen::Vector2d(x - sgn * y_t * std::sin(theta) - 0.5, y_c + sgn * y_t * std::cos(theta));
		}

		bool is_closed() const override { return false; }
	};


	struct JoukowskiAirfoil : public BaseCurve {
		const std::complex<scalar_t> I;    // imaginary number

		scalar_t m;        // approximate maximum camber
		scalar_t t;        // approximate maximum thickness
		scalar_t delta;    // trailing edge slope (angle in degree)

		scalar_t beta;
		scalar_t a;
		scalar_t n;

		std::complex<scalar_t> zeta0;

		scalar_t LE, TE;
		scalar_t chord;

		JoukowskiAirfoil(scalar_t m, scalar_t t, scalar_t delta) :
		    I(0.0, 1.0), m(m), t(t), delta(delta) {

			// center of the unit circle in zeta-plane
			scalar_t xc = -1 / 1.299 * t;
			scalar_t yc = 2.0 * m;
			zeta0.real(xc);
			zeta0.imag(yc);

			beta = std::asin(yc);
			a    = std::cos(beta) + xc;
			n    = 2.0 - delta / 180.0;

			// leading and trailing edge in zeta plane
			std::complex<scalar_t> zeta_le = std::exp(I * (beta + M_PI)) + zeta0;
			std::complex<scalar_t> zeta_te = std::exp(I * (-beta)) + zeta0;

			// leading and trailing edge in z plane
			LE    = std::real(transform(zeta_le));
			TE    = std::real(transform(zeta_te));
			chord = TE - LE;
		}

		std::complex<scalar_t> transform(std::complex<scalar_t> zeta) const {
			return 0.5 * n * a * (std::pow(zeta + a, n) + std::pow(zeta - a, n)) / (std::pow(zeta + a, n) - std::pow(zeta - a, n));
		}

		std::complex<scalar_t> inverse_transform(std::complex<scalar_t> z) const {
			// only work if delta = 0
			z = z * chord + (LE + TE) * 0.5;
			if (std::real(z) > 0)
				return 0.5 * (z + std::sqrt(z * z - 4.0 * a * a));
			else
				return 0.5 * (z - std::sqrt(z * z - 4.0 * a * a));
		}

		Eigen::Vector2d operator()(scalar_t theta) const override {
			std::complex<scalar_t> zeta = std::exp(I * theta) + zeta0;
			std::complex<scalar_t> z    = transform(zeta) - (LE + TE) * 0.5;

			return Eigen::Vector2d(std::real(z) / chord, std::imag(z) / chord);
		}

		bool is_closed() const override { return true; }
	};

}    // namespace dacti::geom
