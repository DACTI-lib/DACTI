#pragma once

#include "integrator/boundary.hpp"
#include "scene/object_mesh.hpp"
#include <filesystem>
#include <optional>
#include <toml++/toml.hpp>

namespace dacti::config {

	template<typename M>
	class Config {
	private:
		using vecn_t = typename M::vecn_t;
		using prim_t = typename M::prim_t;
		/*
			boundary condition function type;
			(coord, normal) -> Boundary
		*/
		using bc_t = std::function<integrator::Boundary<M>(vecn_t, vecn_t)>;

	public:
		std::filesystem::path config_file_path;
		std::filesystem::path config_file_dir;
		std::filesystem::path prj_dir;
		std::string result_out_dir;

		std::string case_name;
		bool visualize_mesh;
		vecn_t active_size;
		scalar_t side_length;

		scalar_t io_interval;
		scalar_t end_time;
		scalar_t signal_time;
		scalar_t convergence_tol;
		scalar_t max_cfl;
		scalar_t max_dt;

		int min_refinement_level;
		int max_refinement_level;
		int max_refinement_level_geom;
		scalar_t refinement_radius;
		vec2_t error_threshold;
		prim_t error_weights;

		prim_t p_ref;
		vecn_t vel_ref;
		std::optional<vec2_t> rho_ref;

		prim_t p_in;    // inflow/Dirichlet boundary condition
		scalar_t fade_in;

		std::array<std::optional<std::array<bc_t, 2>>, M::NumDims> domain_bc;
		std::array<bool, M::NumDims> IsPeriodic;

		struct ic_t {
			// function pointer that takes a position and returns primitive variables
			std::function<prim_t(vecn_t)> x2p;
			std::string type;
			size_t jump_axis;
			scalar_t jump_location;
			bool diffuse_ic;
		};
		ic_t ic;

		struct Object {
			ObjMesh mesh;
			std::string type;
			std::optional<bc_t> bc;

			Object(const ObjMesh &mesh, std::string type, std::optional<bc_t> bc) :
			    mesh(mesh), type(type), bc(bc) {}
		};
		std::vector<Object> objects;

		struct RefineZone {
			vec3_t x0, x1;
			Transform t;
			size_t refinement_level;

			[[nodiscard]] bool contains(const vec3_t &x) const {
				vec3_t x_local = t.transform_point_inverse(x);
				return x_local.cwiseLesserThan(x1).all() && x_local.cwiseGreaterThan(x0).all();
			}
		};
		std::vector<RefineZone> refine_zones;


		// constructor
		Config(const std::string &config_file) {
			config_file_path = std::filesystem::canonical(config_file);
			config_file_dir  = config_file_path.parent_path();
			prj_dir          = config_file_dir.parent_path();

			// load config file
			spdlog::info("Loading case config at '{}'...", config_file_path.string());
			toml::table config_table = toml::parse_file(config_file_path.string());

			load_config(config_table);

			spdlog::info("Case config loaded successfully.");
		}

		void load_config(toml::table &config_table);

		bc_t parse_boundary_condition(toml::table &bc_table);
		ic_t parse_initial_condition(toml::table &ic_table);

		void add_object(const std::vector<ObjMesh> &mesh_group,
		                std::string type,
		                std::optional<bc_t> bc) {
			for (const ObjMesh &mesh: mesh_group) {
				objects.push_back(Object{mesh, type, bc});
			}
		}
	};
}    // namespace dacti::config