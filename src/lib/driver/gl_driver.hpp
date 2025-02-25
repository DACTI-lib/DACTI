#pragma once

#include "core/profile.hpp"


#include <mutex>
#include <omp.h>
#include <spdlog/spdlog.h>


#include "gl/gl.hpp"
#include "gl/glfw.hpp"

#include "mesh_2d/mesh.hpp"
#include "mesh_2d/mesh_scene.hpp"
#include "mesh_2d/parametrized.hpp"

#include <integrator/quadtree_integrator.hpp>
#include <model/euler_2d.hpp>

#include <integrator/error_estimators.hpp>
#include <integrator/slope_limiters.hpp>


#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <thread>

namespace dacti::driver {
	class simulation_output {
		std::vector<float> data;
		int size;
		int frame_index = -1;

	public:
		void store(const auto &integrator) {
			frame_index++;

			size = integrator.n_active_cells();

			data.resize(size * 16);

			integrator.for_each_active_cell([&](int i, const auto &node) {
				const auto &cell = node.cell;

				data[i * 16 + 0] = cell.center[0];
				data[i * 16 + 1] = cell.center[1];

				data[i * 16 + 2] = cell.size;

				for (int j = 0; j < 4; j++) {
					data[i * 16 + 3 + j] = cell.u[j];
					data[i * 16 + 7 + j] = cell.slope[0][j];
					data[i * 16 + 11 + j] = cell.slope[1][j];
				}

				data[i * 16 + 15] = node.acti_level;
			});
		}


		int get_frame_index() {
			return frame_index;
		}

		int get_size() {
			return size;
		}

		void upload_to_gpu(const gl::buffer_object &buffer_object) {
			buffer_object.upload(std::span{data});
		}
	};


	void save_screenshot(glfw::window &window, std::string name, int index) {
		int width, height;

		glfwGetWindowSize(window.ptr(), &width, &height);

		std::vector<char> pixels(width * height * 3);

		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

		std::thread t([pixels = std::move(pixels), width, height, name = std::move(name), index] {
			std::ofstream out(fmt::format("{}_{:010}.ppm", name, index), std::ios::binary | std::ios::out | std::ios::trunc);

			out << "P6\n"
			    << width << " " << height << "\n"
			    << 255 << "\n";

			out.write(pixels.data(), pixels.size());
		});

		t.detach();
	}


	auto ffmpeg(std::string name, int end, int fps) {
		return std::thread([name = std::move(name), end, fps] {
			std::string command = fmt::format("ffmpeg -y -framerate {} -i '{}_%10d.ppm' -frames:v {} -c:v libx265 -crf 25 -vf vflip -pix_fmt yuv420p {}.mp4", fps, name, end + 1, name);

			std::system(command.c_str());


			for (int i = 1; i < end; i++) {
				command = fmt::format("rm {}_{:010}.ppm", name, i);
				std::system(command.c_str());
			}
		});
	}


	struct gl_state {
		glfw::window_config window_config;
		glfw::glfw_context glfw_context;
		glfw::window window;
		gl::vertex_layout layout;
		std::vector<std::tuple<std::string, gl::shader_program>> programs;
		gl::shader_program wireframe_program;
		gl::vertex_array_object vao;
		gl::buffer_object vbo;
		glfw::keyboard_state keyboard_state;
	};
    

	struct gl_driver_config {
		float target_fps = 60.0;
        glfw::window_config window_config;
	};


	template<typename IntegratorConfig, typename Scene, typename Model, typename... Observers>
	class gl_driver {
		using Integrator = typename IntegratorConfig::template impl_t<Model, Scene>;

		std::string simulation_name;

		Integrator integrator;

		Scene scene;

		Model model;

		std::tuple<Observers...> observers;

		simulation_output output;

		gl_driver_config config;

		std::optional<gl_state> gls;


        void write_frame(int io_frame) {
            auto &gls = this->gls.value();

            for (int i = 0; i < gls.programs.size(); i++) {
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				
                auto &[display_program_name, display_program] = gls.programs[i];

				save_screenshot(gls.window, fmt::format("{}_{}", simulation_name, display_program_name), io_frame);
				glUseProgram(gls.wireframe_program.id());
				glDrawArrays(GL_POINTS, 0, output.get_size());
				save_screenshot(gls.window, fmt::format("{}_{}_wf", simulation_name, display_program_name), io_frame);
			}
        }


	public:
		gl_driver(std::string simulation_name, const IntegratorConfig &integrator, const Scene &scene, const Model &model,gl_driver_config config = {}, std::tuple<Observers...> observers = {} )
		    : simulation_name(simulation_name), integrator(integrator, model, scene), scene(scene), model(model), observers(observers), config(config) {
                glfwInit();
			glfw::glfw_context glfw_context;

			glfw::window window{glfw_context, config.window_config};

			gl::vertex_layout layout{
			        {gl::VEC2, gl::FLOAT, gl::VEC4, gl::VEC4, gl::VEC4, gl::FLOAT},
			};

			auto gs = gl::geometry_shader(gl::glsl_source::load_from_file("../res/shader.geom"));
			auto gs_overdraw = gl::geometry_shader(gl::glsl_source::load_from_file("../res/overdraw.geom"));

			auto vs = gl::vertex_shader(gl::glsl_source::load_from_file("../res/shader.vert"));

			auto fs_schlieren = gl::fragment_shader(gl::glsl_source::load_from_file("../res/schlieren.frag"));
			auto fs_overdraw = gl::fragment_shader(gl::glsl_source::load_from_file("../res/overdraw.frag"));
			auto fs_mach = gl::fragment_shader(gl::glsl_source::load_from_file("../res/mach.frag"));
			auto fs_schlieren_acti = gl::fragment_shader(gl::glsl_source::load_from_file("../res/schlieren_acti.frag"));


			auto schlieren_program = gl::shader_program::link(vs, gs, fs_schlieren);
			auto mach_program = gl::shader_program::link(vs, gs, fs_mach);
			auto schlieren_acti_program = gl::shader_program::link(vs, gs, fs_schlieren_acti);

			auto wireframe_overdraw = gl::shader_program::link(vs, gs_overdraw, fs_overdraw);

			auto vao = gl::vertex_array_object{};
			auto vbo = gl::buffer_object{};

			vao.bind_vertex_buffer(layout, vbo);
			vbo.allocate(1 << 30, GL_DYNAMIC_DRAW);

			glClearColor(0.0, 0.0, 0.0, 1.0);

			std::vector<std::tuple<std::string, gl::shader_program>> programs;

			programs.emplace_back("schlieren", std::move(schlieren_program));
			programs.emplace_back("density", std::move(mach_program));
			programs.emplace_back("acti_level", std::move(schlieren_acti_program));

			gls = gl_state{config.window_config, std::move(glfw_context), std::move(window), layout, std::move(programs), std::move(wireframe_overdraw), std::move(vao), std::move(vbo), glfw::keyboard_state{}};

			glfwSetWindowUserPointer(gls->window.ptr(), &gls->keyboard_state);

			glfwSetKeyCallback(gls->window.ptr(), [](GLFWwindow *window, int key, int scancode, int action, int mods) {
				auto &keyboard_state = *static_cast<glfw::keyboard_state *>(glfwGetWindowUserPointer(window));
				keyboard_state.key_callback(window, key, scancode, action, mods);
			});

			glfwSetWindowCloseCallback(gls->window.ptr(), [](GLFWwindow *window) {
				glfwSetWindowShouldClose(window, true);
			});
		}

		void run(double io_interval, std::optional<dacti::scalar_t> end_time = {}) {
			double start_wall = glfwGetTime();
			double previous_frame_time = 0.0;
            double previous_frame_wall = start_wall;

            double io_frame = 0;
            double last_io = 0;

            bool running = true;

            int program_index = 0;

            bool wireframe = false;

            auto &gls = this->gls.value();

            glBindVertexArray(gls.vao.id());

            while (running && !glfwWindowShouldClose(gls.window.ptr()) && (!end_time || integrator.acti_time[0] < *end_time)) {

                integrator.integrate([&](auto &integrator, int acti_level, scalar_t acti_time, int n_substeps, scalar_t dt) {     
                    if(glfwGetTime() - previous_frame_wall < 1.0 / config.target_fps) {
                        return;
                    }

                    // obtain data
                    output.store(integrator);
                    output.upload_to_gpu(gls.vbo);



                    // ###############
                    // # RENDER LOOP #
                    // ###############
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    auto &[display_program_name, display_program] = gls.programs[program_index];

                    glUseProgram(display_program.id());
                    glDrawArrays(GL_POINTS, 0, output.get_size());

                    if(wireframe) {
                        glUseProgram(gls.wireframe_program.id());
                        glDrawArrays(GL_POINTS, 0, output.get_size());
                    }

                    gls.keyboard_state.update();
                    
                    glfwPollEvents();
                    glfwSwapBuffers(gls.window.ptr());
                    // ###############


                    // write frame
                    if (acti_time - last_io > io_interval) {
                        write_frame(io_frame);
                        last_io = acti_time;
                        io_frame++;
                    }

                    // update times
                    double end_wall = glfwGetTime();
                    double elapsed_sim = acti_time - previous_frame_time;
                    double elapsed_wall = end_wall - previous_frame_wall;

                    previous_frame_time = acti_time;
                    previous_frame_wall = end_wall;

                    // print frame info
                    double flux_per_ms = integrator.total_flux_computations / elapsed_sim * 1000.0;
                    integrator.total_flux_computations = 0;
                    
                    fmt::print("Frames Written {}, FPS: {:.1f}, L_max = {}, t = {:.2f} ms, N = {} cells, fluxes/ms = {:.2e}, frame_time = {:.2e} s", io_frame, 1.0 / elapsed_wall, integrator.max_acti_level, acti_time * 1000, integrator.n_active_cells(), flux_per_ms, elapsed_sim);

                    if (end_time) {

                        double remaining_time = *end_time - acti_time;

                        double remaining_wall = remaining_time * elapsed_wall / elapsed_sim;

                        fmt::print(", CPU time rem. ~= {:.2f} s", remaining_wall);
                    }
                    fmt::print("\n");

                    // handle input
                    if(gls.keyboard_state.is_pressed(GLFW_KEY_M)) {
                        program_index = (program_index + 1) % gls.programs.size();
                    }

                    if(gls.keyboard_state.is_pressed(GLFW_KEY_W)) {
                        wireframe = !wireframe;
                    }


                    if (gls.keyboard_state.is_pressed(GLFW_KEY_ESCAPE)) {
                        running = false;
                        glfwSetWindowShouldClose(gls.window.ptr(), true);
                    }
                });
            }

            // TODO: ffmpeg alternative
		}
	};
}// namespace dacti