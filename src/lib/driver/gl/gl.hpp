#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <vector>

#ifdef __APPLE__
	#include <GL/glew.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glew.h>
	#include <GL/gl.h>
#endif

#include <spdlog/spdlog.h>

namespace gl {
	class buffer_object {
		GLuint m_id;

	public:
		// Constructor
		buffer_object() {
			glCreateBuffers(1, &m_id);
		}

		// Prevent Copy
		buffer_object(const buffer_object &) = delete;
		buffer_object &operator=(const buffer_object &) = delete;

		// Enable Move
		buffer_object(buffer_object &&other) noexcept : m_id(other.m_id) {
			other.m_id = 0;// Avoids glDeleteBuffers being called on the same buffer twice
		}

		buffer_object &operator=(buffer_object &&other) noexcept {
			if (this != &other) {
				release();// Release any existing buffer
				std::swap(m_id, other.m_id);
			}
			return *this;
		}

		// Destructor
		~buffer_object() {
			release();
		}

		void allocate(std::size_t size, GLenum usage) const {
			glNamedBufferData(m_id, size, nullptr, usage);
		}

		// Upload data to the buffer
		template<typename T>
		void upload(std::span<T> data, std::size_t offset = 0) const {
			if (offset + data.size_bytes() > size()) {
				throw std::runtime_error{fmt::format("Requested write exceeds buffer object size! ({} > {})",
				                                     offset + data.size_bytes(), size())};
			}

			void *cpu_data = glMapNamedBufferRange(m_id, offset, data.size_bytes(), GL_MAP_WRITE_BIT);
			std::memcpy(cpu_data, data.data(), data.size_bytes());
			glUnmapNamedBuffer(m_id);
		}

		void *map(std::size_t offset, std::size_t length, GLbitfield access_flags) const {
			if (offset + length > size()) {
				throw std::runtime_error{fmt::format("Requested map exceeds buffer object size! ({} > {})",
				                                     offset + length, size())};
			}
			return glMapNamedBufferRange(m_id, offset, length, access_flags);
		}

		void unmap() const {
			glUnmapNamedBuffer(m_id);
		}

		void copy_to(const buffer_object &write_target) const {
			if (size() > write_target.size()) {
				throw std::runtime_error{fmt::format("Requested write exceeds buffer object size! ({} > {})",
				                                     size(), write_target.size())};
			}
			glCopyNamedBufferSubData(m_id, write_target.id(), 0, 0, size());
		}

		// Get the buffer ID
		GLuint id() const {
			return m_id;
		}

		std::size_t size() const {
			GLint64 size;
			glGetNamedBufferParameteri64v(m_id, GL_BUFFER_SIZE, &size);
			return size;
		}

	private:
		// Helper function to delete the buffer
		void release() {
			if (m_id != 0) {
				glDeleteBuffers(1, &m_id);
				m_id = 0;
			}
		}
	};

	struct vertex_attribute_type {
		size_t size;         // Size in bytes of the attribute type
		GLenum type;         // OpenGL type of the attribute
		GLboolean normalized;// Whether the data should be normalized
		size_t dims;         // Number of components in the attribute
	};

	// Common GLSL types as static constants
	constexpr vertex_attribute_type VEC3 = {12, GL_FLOAT, GL_FALSE, 3};
	constexpr vertex_attribute_type VEC4 = {16, GL_FLOAT, GL_FALSE, 4};
	constexpr vertex_attribute_type VEC2 = {8, GL_FLOAT, GL_FALSE, 2};
	constexpr vertex_attribute_type FLOAT = {4, GL_FLOAT, GL_FALSE, 1};
	constexpr vertex_attribute_type INT = {4, GL_INT, GL_FALSE, 1};
	constexpr vertex_attribute_type VEC3I = {12, GL_INT, GL_FALSE, 3};
	constexpr vertex_attribute_type VEC4I = {16, GL_INT, GL_FALSE, 4};
	constexpr vertex_attribute_type VEC2I = {8, GL_INT, GL_FALSE, 2};
	constexpr vertex_attribute_type UNSIGNED_INT = {4, GL_UNSIGNED_INT, GL_FALSE, 1};
	constexpr vertex_attribute_type VEC3UI = {12, GL_UNSIGNED_INT, GL_FALSE, 3};
	constexpr vertex_attribute_type VEC4UI = {16, GL_UNSIGNED_INT, GL_FALSE, 4};
	constexpr vertex_attribute_type VEC2UI = {8, GL_UNSIGNED_INT, GL_FALSE, 2};

	class vertex_layout {
		struct vertex_layout_entry {
			vertex_attribute_type type;
			size_t offset;
		};

		std::vector<vertex_layout_entry> m_entries;
		size_t m_stride;

	public:
		explicit vertex_layout(const std::vector<vertex_attribute_type> &attributes) : m_stride(0) {

			m_entries.resize(attributes.size());

			for (int i = 0; i < attributes.size(); i++) {
				const auto type = attributes[i];
				const size_t rounded_size = (type.size + 3) / 4 * 4;
				m_entries[i].offset = m_stride;
				m_entries[i].type = type;
				m_stride += rounded_size;
			}
		}

		std::span<vertex_layout_entry> entries() {
			return m_entries;
		}

		std::span<const vertex_layout_entry> entries() const {
			return m_entries;
		}

		size_t stride() const {
			return m_stride;
		}
	};

	class vertex_array_object {
		GLuint m_id;

	public:
		// Constructor
		vertex_array_object() {
			glGenVertexArrays(1, &m_id);
		}

		// Prevent Copy
		vertex_array_object(const vertex_array_object &) = delete;
		vertex_array_object &operator=(const vertex_array_object &) = delete;

		// Enable Move
		vertex_array_object(vertex_array_object &&other) noexcept : m_id(other.m_id) {
			other.m_id = 0;// Avoids glDeleteBuffers being called on the same buffer twice
		}

		vertex_array_object &operator=(vertex_array_object &&other) noexcept {
			if (this != &other) {
				release();// Release any existing buffer
				std::swap(m_id, other.m_id);
			}
			return *this;
		}

		GLuint id() const {
			return m_id;
		}

		void bind_vertex_buffer(vertex_layout layout, const buffer_object &vbo) const {
			glBindVertexArray(m_id);
			glBindBuffer(GL_ARRAY_BUFFER, vbo.id());

			for (size_t i = 0; i < layout.entries().size(); i++) {
				const auto entry = layout.entries()[i];

				glVertexAttribPointer(
				        i,
				        entry.type.dims,
				        entry.type.type,
				        entry.type.normalized,
				        layout.stride(),
				        reinterpret_cast<void *>(entry.offset));
				glEnableVertexAttribArray(i);
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		void bind_element_buffer(const buffer_object &ebo) const {
			glBindVertexArray(m_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.id());
			glBindVertexArray(0);
		}

		// Destructor
		~vertex_array_object() {
			release();
		}

	private:
		void release() {
			if (m_id != 0) {
				glDeleteVertexArrays(1, &m_id);
				m_id = 0;
			}
		}
	};

	class spirv_binary {
		std::vector<GLbyte> m_data;

	public:
		static spirv_binary load_from_file(const std::filesystem::path &path) {
			std::ifstream file(path, std::ios::binary | std::ios::ate);
			if (!file) {
				throw std::runtime_error("Failed to open file: " + path.string());
			}

			const size_t file_size = file.tellg();
			std::vector<GLbyte> data(file_size);

			file.seekg(0);
			file.read(reinterpret_cast<char *>(data.data()), file_size);

			return spirv_binary(std::move(data));
		}

		[[nodiscard]] const void *data() const {
			return reinterpret_cast<const void *>(m_data.data());
		}

		[[nodiscard]] GLsizei size() const {
			return m_data.size();
		}

	private:
		[[nodiscard]] spirv_binary(std::vector<GLbyte> &&m_data)
		    : m_data(std::move(m_data)) {}
	};

	class glsl_source {
		std::string m_source;

	public:
		static glsl_source load_from_file(const std::filesystem::path &path) {
			std::ifstream file(path);
			if (!file) {
				throw std::runtime_error("Failed to open file: " + path.string());
			}

			std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
			return glsl_source(std::move(source));
		}

		[[nodiscard]] const char *data() const {
			return m_source.c_str();
		}

		[[nodiscard]] GLsizei size() const {
			return m_source.length();
		}

	private:
		[[nodiscard]] glsl_source(std::string &&source)
		    : m_source(std::move(source)) {}
	};


	class spirv_shader_object {
		GLenum m_type;
		GLuint m_id;

	public:
		spirv_shader_object(const spirv_shader_object &other) = delete;
		spirv_shader_object(spirv_shader_object &&other) noexcept : m_id(other.m_id), m_type(other.m_type) {
			other.m_id = 0;
		}
		spirv_shader_object &operator=(const spirv_shader_object &other) = delete;
		spirv_shader_object &operator=(spirv_shader_object &&other) noexcept {
			if (this != &other) {
				release();// Release any existing buffer
				std::swap(m_id, other.m_id);
				std::swap(m_type, other.m_type);
			}
			return *this;
		}

		[[nodiscard]] explicit spirv_shader_object(const GLenum shader_type, const spirv_binary &binary)
		    : m_type(shader_type) {
			m_id = glCreateShader(shader_type);

			glShaderBinary(1, &m_id, GL_SHADER_BINARY_FORMAT_SPIR_V, binary.data(), binary.size());

			glSpecializeShader(m_id, "main", 0, nullptr, nullptr);


			GLint is_specialized;
			glGetShaderiv(m_id, GL_COMPILE_STATUS, &is_specialized);
			if (is_specialized != GL_TRUE) {
				GLint max_len = 0;
				glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &max_len);

				// The maxLength includes the NULL character
				std::vector<GLchar> info_log(max_len);
				glGetProgramInfoLog(m_id, max_len, &max_len, info_log.data());

				throw std::runtime_error("Shader specialization failed: " + std::string(info_log.data()));
			}
		}


		GLuint id() const {
			return m_id;
		}

	private:
		void release() {
			if (m_id != 0) {
				glDeleteShader(m_id);
				m_id = 0;
			}
		}
	};

	class glsl_shader_object {
		GLenum m_type;
		GLuint m_id;

	public:
		glsl_shader_object(const glsl_shader_object &other) = delete;
		glsl_shader_object(glsl_shader_object &&other) noexcept : m_id(other.m_id), m_type(other.m_type) {
			other.m_id = 0;
		}
		glsl_shader_object &operator=(const glsl_shader_object &other) = delete;
		glsl_shader_object &operator=(glsl_shader_object &&other) noexcept {
			if (this != &other) {
				release();// Release any existing buffer
				std::swap(m_id, other.m_id);
				std::swap(m_type, other.m_type);
			}
			return *this;
		}

		// Constructor adapted for GLSL source shader
		[[nodiscard]] explicit glsl_shader_object(const GLenum shader_type, const std::string &source)
		    : m_type(shader_type) {
			m_id = glCreateShader(shader_type);

			// Set shader source
			const char *src = source.c_str();
			glShaderSource(m_id, 1, &src, nullptr);

			// Compile shader
			glCompileShader(m_id);

			// Check compile status
			GLint is_compiled = 0;
			glGetShaderiv(m_id, GL_COMPILE_STATUS, &is_compiled);
			if (is_compiled == GL_FALSE) {
				GLint max_len = 0;
				glGetShaderiv(m_id, GL_INFO_LOG_LENGTH, &max_len);

				// The maxLength includes the NULL character
				std::vector<GLchar> info_log(max_len);
				glGetShaderInfoLog(m_id, max_len, &max_len, info_log.data());

				throw std::runtime_error("Shader compilation failed: " + std::string(info_log.data()));
			}
		}

		GLuint id() const {
			return m_id;
		}

	private:
		void release() {
			if (m_id != 0) {
				glDeleteShader(m_id);
				m_id = 0;
			}
		}
	};


	template<typename T>
	auto geometry_shader(const T &source);

	template<typename T>
	auto vertex_shader(const T &source);

	template<typename T>
	auto fragment_shader(const T &source);

	template<>
	auto geometry_shader(const spirv_binary &binary) {
		return spirv_shader_object(GL_GEOMETRY_SHADER, binary);
	}

	template<>
	auto vertex_shader(const spirv_binary &binary) {
		return spirv_shader_object(GL_VERTEX_SHADER, binary);
	}

	template<>
	auto fragment_shader(const spirv_binary &binary) {
		return spirv_shader_object(GL_FRAGMENT_SHADER, binary);
	}

	template<>
	auto geometry_shader(const glsl_source &source) {
		return glsl_shader_object(GL_GEOMETRY_SHADER, source.data());
	}

	template<>
	auto vertex_shader(const glsl_source &source) {
		return glsl_shader_object(GL_VERTEX_SHADER, source.data());
	}

	template<>
	auto fragment_shader(const glsl_source &source) {
		return glsl_shader_object(GL_FRAGMENT_SHADER, source.data());
	}


	class shader_program;

	class shader_program_handle {
		GLuint m_id;

		friend class shader_program;

	public:
		auto id() const -> GLuint {
			return m_id;
		}

	private:
		[[nodiscard]] explicit shader_program_handle(const GLuint m_id)
		    : m_id(m_id) {
		}
	};


	class shader_program {
		GLuint m_id;

	public:
		shader_program(const shader_program &other) = delete;
		shader_program(shader_program &&other) noexcept : m_id(other.m_id) {
			other.m_id = 0;
		}

		shader_program &operator=(const shader_program &other) = delete;
		shader_program &operator=(shader_program &&other) noexcept {
			if (this != &other) {
				release();// Release any existing buffer
				std::swap(m_id, other.m_id);
			}
			return *this;
		}

		template<typename ShaderObject>
		static auto link(const ShaderObject &vertex_shader, const ShaderObject &fragment_shader) -> shader_program {
			return shader_program{{vertex_shader.id(), fragment_shader.id()}};
		}

		template<typename ShaderObject>
		static auto link(const ShaderObject &vertex_shader, const ShaderObject &geometry_shader, const ShaderObject &fragment_shader) -> shader_program {
			return shader_program{{vertex_shader.id(), geometry_shader.id(), fragment_shader.id()}};
		}

		GLuint
		id() const {
			return m_id;
		}


		shader_program_handle handle() const {
			return shader_program_handle{m_id};
		}

	private:
		explicit shader_program(const std::initializer_list<GLuint> stages) {
			m_id = glCreateProgram();

			for (const auto &stage_id: stages) {
				glAttachShader(m_id, stage_id);
			}

			glLinkProgram(m_id);

			GLint isLinked = 0;
			glGetProgramiv(m_id, GL_LINK_STATUS, (int *) &isLinked);

			if (isLinked == GL_FALSE) {
				GLint max_len = 0;
				glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &max_len);

				// The maxLength includes the NULL character
				std::vector<GLchar> info_log(max_len);
				glGetProgramInfoLog(m_id, max_len, &max_len, info_log.data());

				// We don't need the program anymore.
				glDeleteProgram(m_id);

				std::string info_log_str{reinterpret_cast<const char *>(info_log.data())};

				spdlog::error("Failed to link shader:\n{}", info_log_str);

				throw std::runtime_error{"Failed to link shader!"};
			}

			for (const auto &stage_id: stages) {
				glDetachShader(m_id, stage_id);
			}
		}


		void release() {
			if (m_id != 0) {
				glDeleteProgram(m_id);
				m_id = 0;
			}
		}
	};
}// namespace gl