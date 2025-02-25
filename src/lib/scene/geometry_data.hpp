#pragma once

#include "config/util.hpp"
#include "geometry.hpp"
#include "igl/readSTL.h"
#include <Eigen/Dense>
#include <fstream>
#include <numbers>

namespace dacti {

	class GeomData {
	private:
		int min_level             = 4;
		int max_level             = 13;
		scalar_t max_error        = 1e-3;
		scalar_t extrusion_height = 2.0;

	public:
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		Eigen::MatrixXd N;
		bool is_smooth = false;

		/**
		 * @brief Constructor for geometry from stl files
		 */
		GeomData(const std::string &filename) {
			std::ifstream in(filename);
			igl::readSTL(in, V, F, N);
		};

		/**
		 * @brief Constructor for build-in 2D geometries
		 *
		 * @param[in] gtable toml table with geometry parameters
		 */
		GeomData(toml::table gtable) {
			std::string prim_type = config::get_entry<std::string>(gtable, "shape");
			spdlog::info("Using built-in geometry: '{}'", prim_type);

			if (prim_type == "ellipse") {
				scalar_t a = config::get_entry<scalar_t>(gtable, "a");
				scalar_t b = config::get_entry<scalar_t>(gtable, "b");

				curve_to_GeomData(geom::Ellipse(a, b),
				                  0.0,
				                  2 * std::numbers::pi);

			} else if (prim_type == "circle") {
				scalar_t r = config::get_entry<scalar_t>(gtable, "radius");

				curve_to_GeomData(geom::Circle(r),
				                  0.0,
				                  2 * std::numbers::pi);

			} else if (prim_type == "naca_airfoil") {
				scalar_t m = config::get_entry<scalar_t>(gtable, "camber");
				scalar_t p = config::get_entry<scalar_t>(gtable, "position");
				scalar_t t = config::get_entry<scalar_t>(gtable, "thickness");

				curve_to_GeomData(geom::NACA4Airfoil(m, p, t),
				                  -1.0,
				                  1.0);

			} else if (prim_type == "Joukowski_airfoil") {
				scalar_t m     = config::get_entry<scalar_t>(gtable, "camber");
				scalar_t t     = config::get_entry<scalar_t>(gtable, "thickness");
				scalar_t delta = config::get_entry<scalar_t>(gtable, "trailing_edge_angle", 0.0);

				curve_to_GeomData(geom::JoukowskiAirfoil(m, t, delta),
				                  0.0,
				                  2 * std::numbers::pi);

			} else if (prim_type == "line") {
				scalar_t length = config::get_entry<scalar_t>(gtable, "length", 0.0);

				Eigen::MatrixXd vertices(2, 2);

				if (length > 0.0) {
					vertices << -length / 2.0, 0,
					    length / 2.0, 0;
				} else {
					vec2_t v1 = config::get_entry<vec2_t>(gtable, "v1");
					vec2_t v2 = config::get_entry<vec2_t>(gtable, "v2");

					vertices << v1[0], v1[1],
					    v2[0], v2[1];
				}

				geom::Polygon line{vertices};

				polygon_to_GeomData(line);
			} else if (prim_type == "diamond") {
				scalar_t a = config::get_entry<scalar_t>(gtable, "width");
				scalar_t b = config::get_entry<scalar_t>(gtable, "height");

				Eigen::MatrixXd vertices(4, 2);

				vertices << 0, b / 2,
				    a / 2, 0,
				    0, -b / 2,
				    -a / 2, 0;

				geom::Polygon diamond{vertices};

				polygon_to_GeomData(diamond);
			} else if (prim_type == "quadrilateral") {
				vec2_t v1 = config::get_entry<vec2_t>(gtable, "v1");
				vec2_t v2 = config::get_entry<vec2_t>(gtable, "v2");
				vec2_t v3 = config::get_entry<vec2_t>(gtable, "v3");
				vec2_t v4 = config::get_entry<vec2_t>(gtable, "v4");

				Eigen::MatrixXd vertices(4, 2);

				vertices << v1[0], v1[1],
				    v2[0], v2[1],
				    v3[0], v3[1],
				    v4[0], v4[1];

				geom::Polygon quadrilateral{vertices};

				polygon_to_GeomData(quadrilateral);
			} else if (prim_type == "triangle") {
				vec2_t v1 = config::get_entry<vec2_t>(gtable, "v1");
				vec2_t v2 = config::get_entry<vec2_t>(gtable, "v2");
				vec2_t v3 = config::get_entry<vec2_t>(gtable, "v3");

				Eigen::MatrixXd vertices(3, 2);

				vertices << v1[0], v1[1],
				    v2[0], v2[1],
				    v3[0], v3[1];

				geom::Polygon triangle{vertices};

				polygon_to_GeomData(triangle);
			} else {
				spdlog::error("Unknown shape type '{}'.", prim_type);
				throw std::runtime_error("Unknown shape type");
			}
		}

		void curve_to_GeomData(const geom::BaseCurve &curve,
		                       scalar_t t_min, scalar_t t_max);

		void polygon_to_GeomData(const geom::Polygon &poly);

		std::vector<scalar_t> compute_adaptive_sampling(const geom::BaseCurve &curve,
		                                                scalar_t t_min, scalar_t t_max);

		void extract_vertices_curve(const geom::BaseCurve &curve,
		                            const std::vector<scalar_t> adaptive_sampling,
		                            Eigen::MatrixXd &V, Eigen::MatrixXd &N);

		void extract_vertices_polygon(const geom::Polygon &poly,
		                              Eigen::MatrixXd &V, Eigen::MatrixXd &N);

		void create_side_faces_curve(const geom::BaseCurve &curve,
		                             Eigen::MatrixXi &F,
		                             size_t num_2d_vertices);

		void create_side_faces_polygon(Eigen::MatrixXi &F, size_t num_2d_vertices);
	};
}    // namespace dacti