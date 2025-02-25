#pragma once

#include "integrator/integrator.hpp"
#include <fmt/core.h>
#include <vtu11/inc/alias.hpp>
#include <vtu11/vtu11.hpp>

namespace dacti::io {

	class vtu_writer {
	private:
		std::string simulation_name;
		std::string out_dir;

	public:
		// constructor
		vtu_writer(std::string &simulation_name, std::string &out_dir) :
		    simulation_name(simulation_name), out_dir(out_dir) {}

		template<typename M, typename L, typename E>
		void write_data(const integrator::Integrator<M, L, E> &integrator,
		                scalar_t t, int frame_index,
		                std::vector<std::vector<scalar_t>> all_FieldVars,
		                std::vector<std::string> all_FieldVar_names) {
			ZoneScoped;

			using I = integrator::Integrator<M, L, E>;

			std::vector<double> points;
			std::vector<double> time;
			std::array<std::vector<double>, M::NumEqns> data_p;
			std::array<std::vector<double>, M::NumParm> data_k;
			std::array<std::vector<double>, I::NumMeta> data_meta;

			std::vector<vtu11::VtkCellType> cell_types;
			std::vector<vtu11::VtkIndexType> point_indices;
			std::vector<vtu11::VtkIndexType> cell_offset;

			int cell_index = 0;

			integrator.for_each_active_cell([&](int i, const auto &node) {
				const auto &cell = node.cell;

				cell_types.push_back(11);
				cell_offset.push_back(cell_index + 8);

				std::array index_order = {0, 1, 3, 2, 4, 5, 7, 6};
				for (int j = 0; j < 8; ++j) {
					vec3_t vertex = cell.get_center().template project<3>() + 0.5 * vec3_t::quadrant(index_order[j]).cwiseMul(cell.get_size().template project_with_default<3>(0.01));

					auto x = vertex.template project<M::NumDims>();

					points.push_back(vertex[0]);
					points.push_back(vertex[1]);
					points.push_back(vertex[2]);

					point_indices.push_back(cell_index + index_order[j]);

					auto p = cell.get_p(x, cell.t);

					for (int k = 0; k < M::NumEqns; ++k) {
						data_p[k].push_back(p[k]);
					}
				}

				for (int k = 0; k < M::NumParm; ++k) {
					data_k[k].push_back(cell.get_k()[k]);
				}

				for (int k = 0; k < I::NumMeta; ++k) {
					data_meta[k].push_back(node.get_metadata()[k]);
				}

				time.push_back(cell.t);

				cell_index += 8;
			});

			std::vector<vtu11::DataSetInfo> datasets;
			std::vector<std::vector<double>> data;


			for (int i = 0; i < M::NumEqns; ++i) {
				datasets.emplace_back(fmt::format("p_{}", i), vtu11::DataSetType::PointData, 1);
				data.emplace_back(data_p[i]);
			}

			for (int i = 0; i < M::NumParm; ++i) {
				datasets.emplace_back(fmt::format("k_{}", i), vtu11::DataSetType::CellData, 1);
				data.emplace_back(data_k[i]);
			}

			for (int i = 0; i < I::NumMeta; ++i) {
				datasets.emplace_back(I::METADATA_NAMES[i], vtu11::DataSetType::CellData, 1);
				data.emplace_back(data_meta[i]);
			}

			for (int i = 0; i < all_FieldVars.size(); ++i) {
				datasets.emplace_back(all_FieldVar_names[i], vtu11::DataSetType::CellData, 1);
				data.emplace_back(all_FieldVars[i]);
			}

			vtu11::Vtu11UnstructuredMesh mesh{points, point_indices, cell_offset, cell_types};

			std::string sim_name = fmt::format("{}/{}_{:04}.vtu", out_dir, simulation_name, frame_index);

			spdlog::info("Wrote file {}", sim_name);

			vtu11::writeVtu(sim_name, mesh, datasets, data, "Base64Inline");
		}
	};
}    // namespace dacti::io
