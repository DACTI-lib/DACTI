#include "object_mesh.hpp"
#include <igl/bfs_orient.h>
#include <igl/loop.h>
#include <igl/orient_outward.h>
#include <igl/orientable_patches.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/unique_edge_map.h>
#include <igl/vertex_components.h>

namespace dacti {
	void normalize_mesh(const Eigen::MatrixXd &V, Transform &t) {
		Eigen::RowVector3d min_corner = V.colwise().minCoeff();
		Eigen::RowVector3d max_corner = V.colwise().maxCoeff();

		Eigen::RowVector3d center_ = (min_corner + max_corner) / 2.0;
		vec3_t center;
		for (int i = 0; i < 3; i++) {
			center[i] = center_[i];
		}

		double max_extent = (max_corner - min_corner).maxCoeff();

		t.normalization /= max_extent;
		t.translation   -= center * t.normalization;
	}

	int split_mesh(const Eigen::MatrixXd &V,
	               const Eigen::MatrixXi &F,
	               std::vector<Eigen::MatrixXd> &V_split,
	               std::vector<Eigen::MatrixXi> &F_split) {
		Eigen::VectorXi C;
		igl::vertex_components(F, C);

		V_split.clear();
		F_split.clear();
		V_split.resize(C.maxCoeff() + 1);
		F_split.resize(C.maxCoeff() + 1);

		for (int component_idx = 0; component_idx < C.maxCoeff() + 1; component_idx++) {
			V_split[component_idx].resize(0, 3);
			F_split[component_idx].resize(0, 3);
		}


		std::vector<size_t> vertex_map(V.rows());

		for (int vertex_idx = 0; vertex_idx < V.rows(); vertex_idx++) {
			int component          = C(vertex_idx);
			vertex_map[vertex_idx] = V_split[component].rows();

			Eigen::MatrixXd &component_vertices = V_split[component];

			size_t new_vertex_idx = component_vertices.rows();


			component_vertices.conservativeResize(new_vertex_idx + 1, 3);
			component_vertices.row(new_vertex_idx) = V.row(vertex_idx);
		}

		for (int face_idx = 0; face_idx < F.rows(); face_idx++) {
			int component = C(F(face_idx, 0));
			assert(C(F(face_idx, 1)) == component && C(F(face_idx, 2)) == component);

			Eigen::MatrixXi &component_faces = F_split[component];

			component_faces.conservativeResize(F_split[component].rows() + 1, 3);

			for (int vertex_idx = 0; vertex_idx < 3; vertex_idx++) {
				component_faces(component_faces.rows() - 1, vertex_idx) = vertex_map[F(face_idx, vertex_idx)];
			}
		}

		return C.maxCoeff() + 1;
	}


	std::vector<ObjMesh> process_geom_data(const GeomData &mesh_data,
	                                       Transform &t,
	                                       bool deduplicate, bool normalize, bool subdivide) {

		Eigen::MatrixXd V_dup = mesh_data.V;
		Eigen::MatrixXi F_dup = mesh_data.F;

		if (normalize) {
			normalize_mesh(V_dup, t);
			V_dup *= t.normalization;
		}

		V_dup = V_dup * t.scale() * t.rotate().transpose();

		V_dup.rowwise() += Eigen::RowVector3d(t.translation[0], t.translation[1], t.translation[2]);

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;

		Eigen::VectorXi _SVI;
		Eigen::VectorXi _SVJ;

		if (deduplicate) {
			igl::remove_duplicate_vertices(V_dup, F_dup, 1e-12, V, _SVI, _SVJ, F);
		} else {
			V = V_dup;
			F = F_dup;
		}

		std::vector<Eigen::MatrixXd> V_split;
		std::vector<Eigen::MatrixXi> F_split;

		int num_components = split_mesh(V, F, V_split, F_split);

		std::vector<ObjMesh> meshes;

		for (int i = 0; i < num_components; i++) {

			Eigen::VectorXi _EMAP;
			Eigen::MatrixXi _E;

			Eigen::VectorXi C;
			Eigen::VectorXi _I;
			Eigen::MatrixXi FF;

			Eigen::MatrixXd V;
			Eigen::MatrixXi F;

			if (subdivide) {
				igl::loop(V_split[i], F_split[i], V, F, 3);
			} else {
				V = V_split[i];
				F = F_split[i];
			}

			igl::orientable_patches(F, C);

			Eigen::VectorXi I_;
			igl::orient_outward(V, F, C, FF, I_);
			F = FF;

			igl::bfs_orient(F, FF, C);

			Eigen::MatrixXd FN;
			Eigen::MatrixXd VN;
			Eigen::MatrixXd EN;
			Eigen::MatrixXi E;
			Eigen::VectorXi EMAP;

			igl::per_face_normals(V, FF, FN);
			igl::per_vertex_normals(V, FF, igl::PerVertexNormalsWeightingType::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, VN);
			igl::per_edge_normals(V, FF, igl::PerEdgeNormalsWeightingType::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, _EMAP);

			igl::unique_edge_map(FF, _E, E, EMAP);

			ObjMesh obj{
			    .V           = V,
			    .F           = FF,
			    .FN          = FN,
			    .EN          = EN,
			    .VN          = VN,
			    .E           = E,
			    .EMAP        = EMAP,
			    .aabb        = igl::AABB<Eigen::MatrixXd, 3>(),
			    .wn_aabb     = std::make_shared<igl::WindingNumberAABB<Eigen::RowVector3d, Eigen::MatrixXd, Eigen::MatrixXi>>(V, F),
			    .fwnb        = std::make_shared<igl::FastWindingNumberBVH>(),
			    .intersector = std::make_shared<igl::embree::EmbreeIntersector>(),
			};

			igl::fast_winding_number(V, FF, 8, *obj.fwnb);

			obj.aabb.init(V, FF);
			obj.intersector->init(V.cast<float>(), FF, true);
			obj.is_smooth = mesh_data.is_smooth;

			meshes.push_back(obj);
		}

		return meshes;
	}
}    // namespace dacti