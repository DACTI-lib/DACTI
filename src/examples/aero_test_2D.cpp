#include "config/config.hpp"
#include "driver/default_driver.hpp"
#include "driver/io/vtu_writer.hpp"
#include "driver/observer/aero_force.hpp"
#include "integrator/error_estimators.hpp"
#include "integrator/integrator.hpp"
#include "integrator/slope_limiters.hpp"
#include "model/euler.hpp"
#include "scene/scene.hpp"


using namespace dacti;

template<typename Model>
void run(const std::string &case_name) {
	auto config = std::make_shared<config::Config<Model>>(case_name);

	scene::Scene<Model> scene(config);

	integrator::Integrator integrator{
	    integrator::slope::minmod{},
	    integrator::error::weighted_laplacian{config->error_weights},
	    scene,
	    config};

	io::vtu_writer writer(config->case_name, config->result_out_dir);

	auto driver = driver::default_driver{
	    integrator,
	    writer,
	    config,
	    std::tuple{observer::AeroForce()}};

	if (config->visualize_mesh) {
		scene.visualize_mesh(integrator.get_domain());
	} else {
		driver.run();
	}
}

int main(int argc, char *argv[]) {

	std::string case_name = argv[1];

	run<model::euler<2>>(case_name);

	return EXIT_SUCCESS;
}
