#pragma once

#include "scene/geometry_data.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <igl/AABB.h>
#include <igl/Hit.h>
#include <igl/barycentric_coordinates.h>
#include <igl/embree/EmbreeIntersector.h>
#include <igl/fast_winding_number.h>
#include <igl/signed_distance.h>

namespace dacti {
	struct ObjMesh {
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		Eigen::MatrixXd FN;
		Eigen::MatrixXd EN;
		Eigen::MatrixXd VN;
		Eigen::MatrixXi E;
		Eigen::VectorXi EMAP;
		igl::AABB<Eigen::MatrixXd, 3> aabb;
		std::shared_ptr<igl::WindingNumberAABB<Eigen::RowVector3d, Eigen::MatrixXd, Eigen::MatrixXi>> wn_aabb;
		std::shared_ptr<igl::FastWindingNumberBVH> fwnb;
		std::shared_ptr<igl::embree::EmbreeIntersector> intersector;
		bool is_smooth;
	};


	struct Transform {
		vec3_t translation     = vec3_t::Zero();
		vec3_t rotation        = vec3_t::Zero();
		vec3_t scaling         = vec3_t::One();
		scalar_t normalization = 1.0;

		[[nodiscard]] Eigen::Matrix3d rotate() const {
			return (Eigen::AngleAxisd(rotation[0] * M_PI / 180.0, Eigen::Vector3d::UnitX()) *
			        Eigen::AngleAxisd(rotation[1] * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
			        Eigen::AngleAxisd(rotation[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ()))
			    .toRotationMatrix();
		}

		[[nodiscard]] Eigen::Matrix3d scale() const {
			return scaling.to_iterable<Eigen::Vector3d>().asDiagonal();
		}

		[[nodiscard]] vec3_t transform_point(const vec3_t &x) const {
			return vec3_t::from_iterable(rotate().transpose() * scale() * x.to_iterable<Eigen::Vector3d>()) + translation;
		}

		[[nodiscard]] vec3_t transform_point_inverse(const vec3_t &x) const {
			return vec3_t::from_iterable(rotate().transpose() * scale().inverse() * (x - translation).to_iterable<Eigen::Vector3d>());
		}
	};


	/** 
     * @brief Compute the signed distance of a point to a 3D mesh
     *
     * @tparam N dimension of the point
	 * @param[in] mesh mesh representation
	 * @param[in] point point to compute the nearest point to
	 * @param[out] nearest nearest point on the mesh
	 * @param[out] normal normal at the nearest point
	 * @return absolute distance to the nearest point
     */
	template<size_t N>
	scalar_t nearest_mesh_point(const ObjMesh &mesh,
	                            const linalg::vec<scalar_t, N> &point,
	                            vec_t<N> &nearest,
	                            vec_t<N> &normal) {

		Eigen::RowVector3d p = Eigen::Vector3d::Zero();

		for (int i = 0; i < N; i++) {
			p[i] = point[i];
		}

		int i;
		Eigen::RowVector3d c;

		double sq_dist = mesh.aabb.squared_distance(mesh.V, mesh.F, p, i, c);

		Eigen::RowVector3d A = mesh.V.row(mesh.F(i, 0));
		Eigen::RowVector3d B = mesh.V.row(mesh.F(i, 1));
		Eigen::RowVector3d C = mesh.V.row(mesh.F(i, 2));
		Eigen::RowVector3d L = Eigen::Vector3d::Zero();

		igl::barycentric_coordinates(c, A, B, C, L);

		Eigen::RowVector3d n_interp = L[0] * mesh.VN.row(mesh.F(i, 0)) + L[1] * mesh.VN.row(mesh.F(i, 1)) + L[2] * mesh.VN.row(mesh.F(i, 2));
		Eigen::RowVector3d n_face   = mesh.FN.row(i);
		Eigen::RowVector3d n;

		if (mesh.is_smooth) {
			n = n_interp;
		} else {
			n = n_face;
		}

		for (int i = 0; i < N; i++) {
			nearest[i] = c[i];
			normal[i]  = n[i];
		}

		return std::sqrt(sq_dist);
	}

	template<size_t N>
	inline bool intersect(const ObjMesh &mesh,
	                      const linalg::vec<scalar_t, N> &_a,
	                      const linalg::vec<scalar_t, N> &_b,
	                      scalar_t &t) {

		Eigen::RowVector3d a = Eigen::Vector3d::Zero();
		Eigen::RowVector3d b = Eigen::Vector3d::Zero();

		for (int i = 0; i < N; i++) {
			a(i) = _a[i];
			b(i) = _b[i];
		}

		Eigen::RowVector3d v = b - a;

		igl::Hit hit;

		if (mesh.intersector->intersectSegment(a.cast<float>(), v.cast<float>(), hit)) {
			t = hit.t;

			return true;
		}

		return false;
	}

	void normalize_mesh(const Eigen::MatrixXd &V, Transform &t);

	int split_mesh(const Eigen::MatrixXd &V,
	               const Eigen::MatrixXi &F,
	               std::vector<Eigen::MatrixXd> &V_split,
	               std::vector<Eigen::MatrixXi> &F_split);

	std::vector<ObjMesh> process_geom_data(const GeomData &geom_data,
	                                       Transform &t,
	                                       bool deduplicate,
	                                       bool normalize = true,
	                                       bool subdivide = false);
}    // namespace dacti