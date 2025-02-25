#include "scene/geometry_data.hpp"
#include <deque>

namespace dacti {
	void GeomData::curve_to_GeomData(const geom::BaseCurve &curve,
	                                 scalar_t t_min, scalar_t t_max) {
		std::vector<scalar_t> adaptive_sampling = compute_adaptive_sampling(curve, t_min, t_max);

		extract_vertices_curve(curve, adaptive_sampling, V, N);
		size_t num_2d_vertices = V.rows() / 2;
		create_side_faces_curve(curve, F, num_2d_vertices);

		is_smooth = true;
	}


	void GeomData::polygon_to_GeomData(const geom::Polygon &poly) {
		extract_vertices_polygon(poly, V, N);
		size_t num_2d_vertices = V.rows() / 2;
		create_side_faces_polygon(F, num_2d_vertices);

		is_smooth = false;
	}


	std::vector<scalar_t> GeomData::compute_adaptive_sampling(const geom::BaseCurve &curve,
	                                                          scalar_t t_min, scalar_t t_max) {
		struct CurveNode {
			scalar_t a;
			size_t left_child  = 0;
			size_t right_child = 0;
			scalar_t length;
			int level;

			CurveNode(scalar_t a, scalar_t length, int level) :
			    a(a), length(length), level(level) {}
		};


		const scalar_t NUM_EPS = 1e-7;
		std::vector<CurveNode> tree{{NAN, NAN, -1}};
		tree.emplace_back(0.5 * (t_min + t_max), t_max - t_min, 0);

		std::deque<size_t> maybe_split;
		maybe_split.push_front(1);

		while (!maybe_split.empty()) {
			size_t i = maybe_split.back();
			maybe_split.pop_back();

			scalar_t a_0 = tree[i].a - tree[i].length / 2.0;
			scalar_t a_1 = tree[i].a + tree[i].length / 2.0;

			int n = 100;

			Eigen::Vector2d min_tangent = Eigen::Vector2d::Constant(std::numeric_limits<scalar_t>::max());
			Eigen::Vector2d max_tangent = Eigen::Vector2d::Constant(std::numeric_limits<scalar_t>::lowest());

			for (int a_i = 0; a_i <= n; a_i++) {
				scalar_t a           = a_0 + (a_1 - a_0) / n * a_i;
				Eigen::Vector2d pn   = curve(a - NUM_EPS);
				Eigen::Vector2d pp   = curve(a + NUM_EPS);
				scalar_t delta_p     = (pp - pn).norm();
				Eigen::Vector2d dpdt = ((pp - pn) / delta_p);
				min_tangent          = min_tangent.cwiseMin(dpdt);
				max_tangent          = max_tangent.cwiseMax(dpdt);
			}

			scalar_t error_est = (max_tangent - min_tangent).norm();

			if (error_est > max_error && tree[i].level < max_level || tree[i].level < min_level) {
				tree[i].left_child  = tree.size();
				tree[i].right_child = tree.size() + 1;

				tree.emplace_back(tree[i].a - 0.25 * tree[i].length, 0.5 * tree[i].length, tree[i].level + 1);
				tree.emplace_back(tree[i].a + 0.25 * tree[i].length, 0.5 * tree[i].length, tree[i].level + 1);

				maybe_split.push_front(tree[i].left_child);
				maybe_split.push_front(tree[i].right_child);
			}
		}

		std::vector<scalar_t> adaptive_sampling{};

		std::function<void(size_t)> add_vertices = [&](size_t index) {
			if (tree[index].left_child && tree[index].right_child) {
				add_vertices(tree[index].left_child);
				add_vertices(tree[index].right_child);
			} else {
				adaptive_sampling.push_back(tree[index].a);
			}
		};

		if (!curve.is_closed()) adaptive_sampling.push_back(t_min);

		add_vertices(1);

		if (!curve.is_closed()) adaptive_sampling.push_back(t_max);

		return adaptive_sampling;
	}


	void GeomData::extract_vertices_curve(const geom::BaseCurve &curve,
	                                      const std::vector<scalar_t> adaptive_sampling,
	                                      Eigen::MatrixXd &V, Eigen::MatrixXd &N) {
		std::vector<Eigen::Vector3d> vertices;
		std::vector<Eigen::Vector3d> normals;
		std::vector<scalar_t> curvatures;

		scalar_t half_height = extrusion_height / 2.0;

		for (int i = 0; i < adaptive_sampling.size(); i++) {
			scalar_t a = adaptive_sampling[i];

			Eigen::Vector2d p  = curve(a);
			Eigen::Vector2d pp = (i == adaptive_sampling.size() - 1 ? p : curve(a + 1e-6));
			Eigen::Vector2d pn = (i == 0 ? p : curve(a - 1e-6));


			vertices.emplace_back(p[0], p[1], -half_height);
			vertices.emplace_back(p[0], p[1], half_height);

			Eigen::Vector2d tangent = pp - pn;

			Eigen::Vector3d normal = Eigen::Vector3d(-tangent[1], tangent[0], 0.0).stableNormalized();

			normals.push_back(normal);
			normals.push_back(normal);
		}

		// Convert to Eigen types
		V = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double *>(vertices.data()), 3, vertices.size()).transpose();
		N = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double *>(normals.data()), 3, normals.size()).transpose();
	}


	void GeomData::extract_vertices_polygon(const geom::Polygon &poly,
	                                        Eigen::MatrixXd &V, Eigen::MatrixXd &N) {
		std::vector<Eigen::Vector3d> vertices;

		scalar_t half_height = extrusion_height / 2.0;

		for (int i = 0; i < poly.vertices.rows(); i++) {
			Eigen::Vector2d v2d = poly.vertices.row(i);

			vertices.push_back(Eigen::Vector3d(v2d[0], v2d[1], -half_height));
			vertices.push_back(Eigen::Vector3d(v2d[0], v2d[1], half_height));
		}

		V = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double *>(vertices.data()), 3, vertices.size()).transpose();
		N = Eigen::MatrixXd::Zero(V.rows(), 3);
	}


	void GeomData::create_side_faces_curve(const geom::BaseCurve &curve, Eigen::MatrixXi &F, size_t num_2d_vertices) {
		size_t n = num_2d_vertices;

		if (!curve.is_closed())
			n--;

		F.resize(n * 2, 3);

		for (size_t i = 0; i < n; i++) {
			size_t next = (i + 1) % num_2d_vertices;

			F.row(i * 2) << i * 2, next * 2, i * 2 + 1;
			F.row(i * 2 + 1) << next * 2, next * 2 + 1, i * 2 + 1;
		}
	}


	void GeomData::create_side_faces_polygon(Eigen::MatrixXi &F, size_t num_2d_vertices) {
		F.resize(num_2d_vertices * 2, 3);
		for (size_t i = 0; i < num_2d_vertices; i++) {
			size_t next = (i + 1) % num_2d_vertices;

			F.row(i * 2) << i * 2, next * 2, i * 2 + 1;
			F.row(i * 2 + 1) << next * 2, next * 2 + 1, i * 2 + 1;
		}
	}
}    // namespace dacti