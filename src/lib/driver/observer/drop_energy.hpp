#pragma once

#include "config/config.hpp"
#include "driver/observer/common.hpp"

namespace dacti::observer {

	class DropEnergy : public Observer {
	public:
		DropEnergy() :
		    Observer("Droplet Energy", {"ke", "Etot"}) {}

		template<typename Integrator, typename Model>
		void observe(const Integrator &integrator,
		             const std::shared_ptr<config::Config<Model>> config,
		             const bool observe_FieldVars = true) const {
			scalar_t ke   = 0;
			scalar_t Etot = 0;

			integrator.for_each_active_cell([&](int i, auto &node) {
				scalar_t alpha = node.cell.get_p()[Model::ia];
				scalar_t vol   = node.cell.volume();
				scalar_t rho   = node.cell.get_p()[0] + node.cell.get_p()[1];
				auto velocity  = node.cell.get_velocity();
				scalar_t norm  = std::sqrt(node.cell.get_gradient()[0][Model::da] * node.cell.get_gradient()[0][Model::da] +
                                  node.cell.get_gradient()[1][Model::da] * node.cell.get_gradient()[1][Model::da]);

				ke   += vol * 0.5 * rho * velocity.squaredNorm();
				Etot += vol * (0.5 * rho * velocity.squaredNorm() + Model::sigma * norm);
			});

			_IntegVars = {ke, Etot};
		}
	};
}    // namespace dacti::observer