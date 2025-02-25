#include "config/config.hpp"
#include "config/util.hpp"
#include "model/euler.hpp"
#include "model/multi.hpp"
#include "model/navier_stokes.hpp"
#include <igl/writeSTL.h>

namespace dacti::config {
	template<typename M>
	void Config<M>::load_config(toml::table &config_table) {
		case_name                 = get_entry<std::string>(config_table, "scene.name");
		active_size               = get_entry<vecn_t>(config_table, "scene.domain_size");
		side_length               = active_size.maxCoeff();
		io_interval               = get_entry<scalar_t>(config_table, "run.print_interval");
		end_time                  = get_entry<scalar_t>(config_table, "run.end_time");
		signal_time               = get_entry<scalar_t>(config_table, "run.signal_time", -1.0);
		convergence_tol           = get_entry<scalar_t>(config_table, "run.steady_tol", -1.0);
		max_cfl                   = get_entry<scalar_t>(config_table, "integrator.max_cfl");
		max_dt                    = get_entry<scalar_t>(config_table, "integrator.max_dt", io_interval);
		min_refinement_level      = get_entry<int>(config_table, "integrator.min_refinement_level");
		max_refinement_level      = get_entry<int>(config_table, "integrator.max_refinement_level");
		max_refinement_level_geom = get_entry<int>(config_table, "integrator.max_refinement_level_geom", max_refinement_level);
		refinement_radius         = get_entry<scalar_t>(config_table, "integrator.refinement_radius", 2.0);
		error_threshold           = get_entry<vec2_t>(config_table, "integrator.error_threshold");
		error_weights             = get_entry<prim_t>(config_table, "integrator.error_weights", prim_t::One());

		if (config_table["scene"]["visualize_mesh"])
			visualize_mesh = config_table["scene"]["visualize_mesh"].as_boolean()->get();

		// create output directory for results
		result_out_dir = prj_dir / "out" / case_name;
		if (!std::filesystem::exists(prj_dir / "out")) {
			std::filesystem::create_directory(prj_dir / "out");
			std::filesystem::create_directory(result_out_dir);
		} else {
			std::filesystem::create_directory(result_out_dir);
		}
		spdlog::info("Results output directory: '{}'", result_out_dir);

		// load reference values
		M::dict(config_table["model"], p_ref, vel_ref, rho_ref);

		// load initial conditions
		ic = parse_initial_condition(*config_table["initial_condition"].as_table());

		// load domain boundary conditions
		std::array<std::string, 3> DIM_NAMES = {"X", "Y", "Z"};
		for (int dim = 0; dim < M::NumDims; dim++) {
			if (auto type = config_table["boundary_conditions"][DIM_NAMES[dim]]["type"].value<std::string>()) {
				if (*type == "periodic") {
					IsPeriodic[dim] = true;
				} else {
					IsPeriodic[dim] = false;
				}
			} else {
				bc_t neg = parse_boundary_condition(*config_table["boundary_conditions"][DIM_NAMES[dim]]["neg"].as_table());
				bc_t pos = parse_boundary_condition(*config_table["boundary_conditions"][DIM_NAMES[dim]]["pos"].as_table());

				domain_bc[dim] = {neg, pos};
			}
		}

		// Load geometries and prepare for meshing
		std::unordered_map<std::string, Transform> transforms;

		if (config_table.contains("geometry")) {
			config_table["geometry"].as_array()->for_each([&](toml::table geom) {
				std::string geom_name = geom["name"].as_string()->get();
				std::string geom_type = get_entry<std::string>(geom, "type", "solid");

				std::vector<ObjMesh> mesh_group;

				Transform t;
				t.rotation    = get_entry<vec3_t>(geom, "rotation", vec3_t::Zero());
				t.scaling     = get_entry<vec3_t>(geom, "scaling", vec3_t::One());
				t.translation = get_entry<vec3_t>(geom, "translation", vec3_t::Zero());

				if (geom["path"]) {
					std::string path                = geom["path"].as_string()->get();
					std::filesystem::path geom_path = config_file_dir / path;
					std::string geom_file           = std::filesystem::canonical(geom_path).string();

					spdlog::info("Loading geometry file at '{}'...", geom_file);

					GeomData geom_data(geom_file);
					geom_data.is_smooth = get_entry<bool>(*geom.as_table(), "smooth", true);

					mesh_group = process_geom_data(geom_data, t, true);

					spdlog::info("Loading boundary mesh {}...", geom_path.string());
					spdlog::info("Contained {} objects.", mesh_group.size());

				} else if (geom["shape"]) {    // built-in 2D Geometries
					GeomData geom_data(*geom.as_table());
					mesh_group = process_geom_data(geom_data, t, false, false);
				} else {
					spdlog::error("Geometry '{}' needs to either have an associated file path or a shape.");
					throw std::runtime_error("Undefined Geometry");
				}

				// load geometry boundary conditions
				std::optional<bc_t> bc = std::nullopt;
				if (geom_type == "solid" && !config_table["boundary_conditions"][geom_name]) {
					spdlog::error("Must define boundary condition for solid geometry '{}' ", geom_name);
					throw std::runtime_error("Require boundary condition");
				} else if (config_table["boundary_conditions"][geom_name]) {
					bc = parse_boundary_condition(*config_table["boundary_conditions"][geom_name].as_table());
				}

				add_object(mesh_group, geom_type, bc);

#ifdef NDEBUG
				for (int i = 0; auto mesh: mesh_group) {
					igl::writeSTL(fmt::format("{}/{}_component_{}.stl", result_out_dir, geom_name, i), mesh.V, mesh.F, mesh.VN);
					i++;
				}
#endif

				if (transforms.contains(geom_name)) {
					spdlog::warn("Duplicate Object name: {}", geom_name);
				}

				transforms[geom_name] = t;
			});
		}

		// load refinement zones
		if (config_table.contains("refinement_zone")) {
			config_table["refinement_zone"].as_array()->for_each([&](toml::table zone) {
				Transform t;
				if (zone["transform_like"])
					t = transforms[zone["transform_like"].as_string()->get()];

				refine_zones.emplace_back(
				    get_entry<vec3_t>(zone, "start"),
				    get_entry<vec3_t>(zone, "end"),
				    t,
				    get_entry<size_t>(zone, "max_refinement_level"));
			});
		}
	}


	template<typename M>
	Config<M>::bc_t Config<M>::parse_boundary_condition(toml::table &bc_table) {
		std::string type = get_entry<std::string>(bc_table, "type");

		if (type == "noslip") {
			return [](vecn_t x, vecn_t n) {
				return integrator::Boundary<M>::noslip(n);
			};
		}

		if (type == "freeslip") {
			return [](vecn_t x, vecn_t n) {
				return integrator::Boundary<M>::freeslip(n);
			};
		}

		if (type == "inflow") {
			fade_in = get_entry<scalar_t>(bc_table, "fade_in", 0.0);

			return [r = p_ref, fade_in = fade_in](vecn_t x, vecn_t n) {
				return integrator::Boundary<M>::inflow(r, fade_in);
			};
		}

		if (type == "inflow_r") {
			fade_in = get_entry<scalar_t>(bc_table, "fade_in", 0.0);
			p_in    = get_entry<prim_t>(bc_table, "value");

			return [r = p_in, fade_in = fade_in](vecn_t x, vecn_t n) {
				return integrator::Boundary<M>::inflow_r(r, fade_in);
			};
		}

		if (type == "outflow") {
			return [r = p_ref](vecn_t x, vecn_t n) {
				return integrator::Boundary<M>::outflow(r[M::ip]);
			};
		}

		if (type == "zeroGradient") {
			return [r = p_ref](vecn_t x, vecn_t n) {
				return integrator::Boundary<M>::zeroGradient();
			};
		}

		throw std::runtime_error{"illegal bc type"};
	}


	template<typename M>
	Config<M>::ic_t Config<M>::parse_initial_condition(toml::table &ic_table) {
		std::string type = get_entry<std::string>(ic_table, "type");

		bool diffuse_ic = false;
		if (ic_table["initial_condition"]["diffuse_ic"])
			diffuse_ic = ic_table["initial_condition"]["diffuse_ic"].as_boolean()->get();

		if (type == "uniform") {

			std::string value = get_entry<std::string>(ic_table, "value", "reference");

			if (value == "reference") {
				return {[r = p_ref](vecn_t x) { return r; }, type, 0, 0.0, diffuse_ic};
			}
			if (value == "fixed") {
				prim_t p_fixed = get_entry<prim_t>(ic_table, "fixed_value");
				return {[r = p_ref](vecn_t x) { return r; }, type, 0, 0.0, diffuse_ic};
			}

			spdlog::error("Unknown uniform value type "
			              "{}"
			              "",
			              value);
		}

		if (type == "jump") {
			size_t dim        = get_entry<size_t>(ic_table, "axis");
			scalar_t location = get_entry<scalar_t>(ic_table, "location");
			prim_t value_neg  = get_entry<prim_t>(ic_table, "value_neg");
			prim_t value_pos  = get_entry<prim_t>(ic_table, "value_pos");

			return {
			    [r_neg = value_neg, r_pos = value_pos, loc = location, dim = dim](vecn_t x) {
				    if (x[dim] > loc)
					    return r_pos;
				    else
					    return r_neg;
			    },
			    type,
			    dim,
			    location,
			    diffuse_ic};
		}

		spdlog::error("Unknown initial condition type "
		              "{}"
		              "",
		              type);
		throw std::runtime_error("illegal arg");
	}
}    // namespace dacti::config

template class dacti::config::Config<dacti::model::euler<1>>;
template class dacti::config::Config<dacti::model::euler<2>>;
template class dacti::config::Config<dacti::model::euler<3>>;
template class dacti::config::Config<dacti::model::navier_stokes<1>>;
template class dacti::config::Config<dacti::model::navier_stokes<2>>;
template class dacti::config::Config<dacti::model::navier_stokes<3>>;
template class dacti::config::Config<dacti::model::multi<1>>;
template class dacti::config::Config<dacti::model::multi<2>>;
template class dacti::config::Config<dacti::model::multi<3>>;