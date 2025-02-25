#pragma once

#include <spdlog/spdlog.h>

#include "gl.hpp"
#include <GLFW/glfw3.h>



#include <unordered_set>


struct __callbacks {

	static void glfw_error_callback(int error, const char *description) {
		spdlog::error("[GLFW] {}", error, description);
	}

	static void gl_debug_message_callback(GLenum source,
	                                      GLenum type,
	                                      GLuint id,
	                                      GLenum severity,
	                                      GLsizei length,
	                                      const GLchar *message,
	                                      const void *userParam) {
		// Determine the level of severity and use the corresponding spdlog function
		spdlog::level::level_enum log_level;
		switch (severity) {
			case GL_DEBUG_SEVERITY_HIGH:
				log_level = spdlog::level::critical;
				break;
			case GL_DEBUG_SEVERITY_MEDIUM:
				log_level = spdlog::level::err;
				break;
			case GL_DEBUG_SEVERITY_LOW:
				log_level = spdlog::level::warn;
				break;
			case GL_DEBUG_SEVERITY_NOTIFICATION:
				log_level = spdlog::level::info;
				break;
			default:
				log_level = spdlog::level::trace;
				break;
		}

		// Log the message
		spdlog::log(log_level, fmt::format("[OpenGL Debug] message = {}", message));
	}
};

namespace glfw {

	struct window_config {
		int width = 1280;
		int height = 720;
		std::string title = "GLFW Window";
		bool fullscreen = false;
		bool resizable = false;
		bool decorated = true;
		bool vsync = true;
		int opengl_major = 4;
		int opengl_minor = 0;
		int msaa_samples = 4;
	};

	class glfw_context {
		public:
		glfw_context() {
			
		}



		~glfw_context() {
			
		}
	};

	class keyboard_state {
		std::unordered_set<int> m_keys_pressed;
		std::unordered_set<int> m_keys_down;
		std::unordered_set<int> m_keys_released;


	public:
		void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
			if (action == GLFW_PRESS) {
				m_keys_pressed.insert(key);
				m_keys_down.insert(key);
			} else if (action == GLFW_RELEASE) {
				m_keys_down.erase(key);
				m_keys_released.insert(key);
			}
		}

		void update() {
			m_keys_pressed.clear();
			m_keys_released.clear();
		}


		bool is_down(int key) const {
			return m_keys_down.contains(key);
		}

		bool is_pressed(int key) const {
			return m_keys_pressed.contains(key);
		}

		bool is_released(int key) const {
			return m_keys_released.contains(key);
		}
	};

	class Iapplication {
	public:
		virtual void update(GLFWwindow *window){};

		virtual void init(GLFWwindow *window) {}

		virtual void on_key_event(GLFWwindow *window, int key, int scancode, int action, int mods) {}

		virtual void on_cursor_pos_event(GLFWwindow *window, double xpos, double ypos) {}

		virtual void on_mouse_button_event(GLFWwindow *window, int button, int action, int mods) {}

		virtual void on_scroll_event(GLFWwindow *window, double xoffset, double yoffset) {}

		virtual bool should_close() {
			return false;
		}
	};

	class window {
		GLFWwindow *m_window;

	public:
		window(const glfw_context &ctx, const window_config &config) {
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, config.opengl_major);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, config.opengl_minor);
			glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
			glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

			glfwWindowHint(GLFW_RESIZABLE, config.resizable);
			glfwWindowHint(GLFW_DECORATED, config.decorated);
			glfwWindowHint(GLFW_VISIBLE, false);
			glfwWindowHint(GLFW_SAMPLES, config.msaa_samples);

			m_window = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);

			if (!m_window) {
				spdlog::error("Failed to create GLFW window");
				glfwTerminate();
				throw std::runtime_error("Failed to create GLFW window");
			}

			glfwMakeContextCurrent(m_window);


			if (config.vsync) {
				glfwSwapInterval(1);
			}

			glewExperimental = GL_TRUE;
			glewInit();

			glfwShowWindow(m_window);

			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
			glEnable(GL_DEBUG_OUTPUT);
			glDebugMessageCallback(__callbacks::gl_debug_message_callback, nullptr);
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
		}

		window(const window &) = delete;
		window(window &&other) {
			m_window = other.m_window;
			other.m_window = nullptr;
		}

		window &operator=(const window &) = delete;
		window &operator=(window &&other) {
			m_window = other.m_window;
			other.m_window = nullptr;
			return *this;
		}

		void start_application(Iapplication *application) {
			application->init(m_window);

			// register callbacks

			glfwSetWindowUserPointer(m_window, application);

			glfwSetKeyCallback(m_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
				auto app = reinterpret_cast<Iapplication *>(glfwGetWindowUserPointer(window));
				app->on_key_event(window, key, scancode, action, mods);
			});

			glfwSetCursorPosCallback(m_window, [](GLFWwindow *window, double xpos, double ypos) {
				auto app = reinterpret_cast<Iapplication *>(glfwGetWindowUserPointer(window));
				app->on_cursor_pos_event(window, xpos, ypos);
			});

			glfwSetMouseButtonCallback(m_window, [](GLFWwindow *window, int button, int action, int mods) {
				auto app = reinterpret_cast<Iapplication *>(glfwGetWindowUserPointer(window));
				app->on_mouse_button_event(window, button, action, mods);
			});

			glfwSetScrollCallback(m_window, [](GLFWwindow *window, double xoffset, double yoffset) {
				auto app = reinterpret_cast<Iapplication *>(glfwGetWindowUserPointer(window));
				app->on_scroll_event(window, xoffset, yoffset);
			});

			while (!glfwWindowShouldClose(m_window) && !application->should_close()) {
				glClear( GL_COLOR_BUFFER_BIT );

				application->update(m_window);

				glfwPollEvents();
				glfwSwapBuffers(m_window);
			}

			glfwSetWindowUserPointer(m_window, nullptr);
		}

		~window() {
			if (m_window) {
				glfwDestroyWindow(m_window);
			}
		}

		GLFWwindow *ptr() {
			return m_window;
		}
	};
}// namespace glfw