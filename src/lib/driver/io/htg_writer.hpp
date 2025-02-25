#pragma once

#include "integrator/integrator.hpp"
#include <fmt/core.h>
#include <vtkBitArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkHyperTreeGrid.h>
#include <vtkHyperTreeGridGeometry.h>
#include <vtkHyperTreeGridNonOrientedCursor.h>
#include <vtkNew.h>
#include <vtkXMLHyperTreeGridWriter.h>


namespace dacti::io {

	class htg_writer {
	private:
		std::string simulation_name;
		std::string out_dir;

	public:
		// constructor
		htg_writer(std::string &simulation_name, std::string &out_dir) :
		    simulation_name(simulation_name), out_dir(out_dir) {}

		template<typename M, typename L, typename E>
		void write_data(
		    const integrator::Integrator<M, L, E> &integrator,
		    scalar_t t,
		    int frame_index,
		    std::vector<std::vector<scalar_t>> injections,
		    std::vector<std::string> injection_names) {
			ZoneScoped;

			using I                 = integrator::Integrator<M, L, E>;
			const size_t root_index = integrator.domain.root_index;

			vtkNew<vtkHyperTreeGrid> htg;
			htg->SetBranchFactor(2);

			// define base grid with only one sqaure cell (two vertices in each dim)
			if constexpr (I::NumDims == 1) {
				htg->SetDimensions(2, 1, 1);
			} else if constexpr (I::NumDims == 2) {
				htg->SetDimensions(2, 2, 1);
			} else if constexpr (I::NumDims == 3) {
				htg->SetDimensions(2, 2, 2);
			}

			vtkNew<vtkDoubleArray> x_base, y_base, z_base;
			auto domain_size = integrator.config->active_size;
			x_base->SetNumberOfValues(2);
			x_base->SetValue(0, -0.5 * domain_size[0]);    // x_min
			x_base->SetValue(1, 0.5 * domain_size[0]);     // x_max
			y_base->SetNumberOfValues(2);
			y_base->SetValue(0, -0.5 * domain_size[0]);    // y_min
			y_base->SetValue(1, 0.5 * domain_size[0]);     // y_max
			z_base->SetNumberOfValues(2);
			z_base->SetValue(0, -0.5 * domain_size[0]);    // z_min
			z_base->SetValue(1, 0.5 * domain_size[0]);     // z_max

			htg->SetXCoordinates(x_base);
			htg->SetYCoordinates(y_base);
			htg->SetZCoordinates(z_base);

			/*
				*	Define data arrays
				*/
			std::vector<vtkSmartPointer<vtkDoubleArray>> data_p;
			for (int i = 0; i < M::NumEqns; ++i) {
				vtkSmartPointer<vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();

				std::string scalar_name = fmt::format("p_{}", i);

				array->SetName(scalar_name.c_str());
				array->SetNumberOfValues(0);
				htg->GetCellData()->AddArray(array);
				htg->GetCellData()->SetActiveScalars(scalar_name.c_str());

				data_p.emplace_back(array);
			}

			std::vector<vtkSmartPointer<vtkDoubleArray>> data_meta;
			for (int i = 0; i < I::NumMeta; ++i) {
				vtkSmartPointer<vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();

				array->SetName(I::METADATA_NAMES[i]);
				array->SetNumberOfValues(0);
				htg->GetCellData()->AddArray(array);
				htg->GetCellData()->SetActiveScalars(I::METADATA_NAMES[i]);

				data_meta.emplace_back(array);
			}

			/*
				*	Define a binary material mask for inactive cells
				*	True: hide cell; False: show cell
				*/
			vtkNew<vtkBitArray> maskarray;
			maskarray->SetName("mask");
			maskarray->SetNumberOfValues(0);
			maskarray->InsertTuple1(root_index, false);

			/*
				*	Construct the hypertree grid
				*/
			vtkNew<vtkHyperTreeGridNonOrientedCursor> cursor;
			htg->InitializeNonOrientedCursor(cursor, 0, true);
			cursor->SetGlobalIndexStart(root_index);

			const auto tree = integrator.get_nodes();

			std::array<std::vector<size_t>, I::MAX_LEVEL> split_node_sets;
			assign_to_split_node_sets(tree, split_node_sets, root_index);

			for (int level = 0; level < I::MAX_LEVEL; level++) {

				if (split_node_sets[level].empty()) break;

				std::vector<size_t> node_set = split_node_sets[level];    // nodes to be split

				// split node
				for (size_t index: node_set) {
					// move cursor to target node
					std::vector<size_t> path;
					get_path_to_root(tree, index, path);

					for (int i = path.size() - 1; i >= 0; i--) {
						cursor->ToChild(path[i]);
					}

					// split
					cursor->SubdivideLeaf();

					// check if child is a leaf and write data
					for (size_t i = 0; i < (1 << I::NumDims); i++) {
						size_t child_index     = tree[index].child_indices[i];
						const auto &child_node = tree[child_index];

						cursor->ToChild(i);
						size_t idx = cursor->GetGlobalNodeIndex();
						maskarray->InsertTuple1(idx, false);

						if (child_node.is_leaf) {
							// TODO: maybe not write inactive cell values
							for (int j = 0; j < M::NumEqns; j++) {
								data_p[j]->InsertTuple1(idx, child_node.cell.get_p()[j]);
							}
							for (int j = 0; j < I::NumMeta; j++) {
								data_meta[j]->InsertTuple1(idx, child_node.get_metadata()[j]);
							}

							if (!child_node.is_active)
								maskarray->InsertTuple1(idx, true);
						}
						cursor->ToParent();
					}
					// back to root
					cursor->ToRoot();
				}
			}
			htg->SetMask(maskarray);

			/*
				*	write hypergrid to file
				*/
			vtkNew<vtkXMLHyperTreeGridWriter> writer;
			std::string file_name = fmt::format("{}/{}_{:04}.htg", out_dir, simulation_name, frame_index);
			writer->SetFileName(file_name.c_str());
			writer->SetInputData(htg);
			writer->Write();
			spdlog::info("Wrote file {}", file_name);
		}

		void assign_to_split_node_sets(const auto &tree, auto &split_node_sets, size_t root_index) {
			for (int i = root_index; i < tree.size(); ++i) {
				auto &node = tree[i];
				if (!node.is_leaf) {
					split_node_sets[node.tree_level].push_back(i);
				}
			}
		}

		void get_path_to_root(const auto &tree, size_t index, std::vector<size_t> &path) {
			size_t child_index  = index;
			size_t parent_index = tree[index].parent_index;

			for (int level = tree[index].tree_level; level > 0; --level) {
				path.push_back(tree[parent_index].find_child(child_index));
				child_index  = parent_index;
				parent_index = tree[parent_index].parent_index;
			}
		}
	};
}    // namespace dacti::io
