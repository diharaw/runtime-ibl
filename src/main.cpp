#include <application.h>
#include <mesh.h>
#include <camera.h>
#include <material.h>
#include <memory>
#include <iostream>
#include <stack>
#include <random>
#include <chrono>


#define CAMERA_FAR_PLANE 10000.0f
#define ENVIRONMENT_MAP_SIZE 512
#define PREFILTER_MAP_SIZE 256
#define PREFILTER_MIP_LEVELS 7
#define IRRADIANCE_CUBEMAP_SIZE 128
#define IRRADIANCE_WORK_GROUP_SIZE 8
#define SH_INTERMEDIATE_SIZE (IRRADIANCE_CUBEMAP_SIZE/IRRADIANCE_WORK_GROUP_SIZE)

class RuntimeIBL : public dw::Application
{
protected:
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
	bool init(int argc, const char* argv[]) override
	{
		// Create GPU resources.
		if (!create_shaders())
			return false;

		// Load mesh.
		if (!load_mesh())
			return false;

		if (!load_environment_map())
			return false;

		if (!create_framebuffer())
			return false;

		// Create camera.
		create_camera();
		create_cube();
		convert_env_map();

		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void update(double delta) override
	{
		// Update camera.
		update_camera();

		if (m_show_gui)
			ui();

		project_cubemap_sh();

		render_meshes();

		render_cubemap();

		if (m_debug_mode)
			m_debug_draw.frustum(m_main_camera->m_view_projection, glm::vec3(0.0f, 1.0f, 0.0f));

        // Render debug draw.
        m_debug_draw.render(nullptr, m_width, m_height, m_debug_mode ? m_debug_camera->m_view_projection : m_main_camera->m_view_projection);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void shutdown() override
	{
		dw::Mesh::unload(m_mesh);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void window_resized(int width, int height) override
	{
		// Override window resized method to update camera projection.
		m_main_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));
        m_debug_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height));

		create_framebuffer();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
    
    void key_pressed(int code) override
    {
        // Handle forward movement.
        if(code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if(code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;
        
        // Handle sideways movement.
        if(code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if(code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;

		if (code == GLFW_KEY_K)
			m_debug_mode = !m_debug_mode;
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void key_released(int code) override
    {
        // Handle forward movement.
        if(code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;
        
        // Handle sideways movement.
        if(code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;

		if (code == GLFW_KEY_G)
			m_show_gui = !m_show_gui;
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------

	void mouse_pressed(int code) override
	{
		// Enable mouse look.
		if (code == GLFW_MOUSE_BUTTON_RIGHT)
			m_mouse_look = true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void mouse_released(int code) override
	{
		// Disable mouse look.
		if (code == GLFW_MOUSE_BUTTON_RIGHT)
			m_mouse_look = false;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

protected:

	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::AppSettings intial_app_settings() override
	{
		dw::AppSettings settings;

		settings.resizable = true;
		settings.maximized = false;
		settings.refresh_rate = 60;
		settings.major_ver = 4;
		settings.width = 1280;
		settings.height = 720;
		settings.title = "Runtime IBL";

		return settings;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

private:

	// -----------------------------------------------------------------------------------------------------------------------------------
	
	void ui()
	{
		static const char* items[] = { "Environment Map", "Irradiance", "Prefiltered" };

		if (ImGui::BeginCombo("Skybox", items[m_type], 0))
		{
			for (int i = 0; i < 3; i++)
			{
				bool is_selected = (m_type == i);

				if (ImGui::Selectable(items[i], is_selected))
					m_type = i;
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}

			ImGui::EndCombo();
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
	
	bool load_environment_map()
	{
		m_env_map = std::unique_ptr<dw::Texture2D>(dw::Texture2D::create_from_files("Arches_E_PineTree_3k.hdr", true, false));
		m_env_map->set_min_filter(GL_LINEAR);
		m_env_map->set_mag_filter(GL_LINEAR);
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_shaders()
	{
		{
			// Create general shaders
			m_cubemap_convert_vs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/equirectangular_to_cubemap_vs.glsl"));
			m_cubemap_convert_fs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/equirectangular_to_cubemap_fs.glsl"));

			if (!m_cubemap_convert_vs || !m_cubemap_convert_fs)
			{
				DW_LOG_FATAL("Failed to create Shaders");
				return false;
			}

			// Create general shader program
			dw::Shader* shaders[] = { m_cubemap_convert_vs.get(), m_cubemap_convert_fs.get() };
			m_cubemap_convert_program = std::make_unique<dw::Program>(2, shaders);

			if (!m_cubemap_convert_program)
			{
				DW_LOG_FATAL("Failed to create Shader Program");
				return false;
			}
		}

		//{
		//	// Create general shaders
		//	m_prefilter_cs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_COMPUTE_SHADER, "shader/prefilter_cs.glsl"));

		//	if (!m_prefilter_cs)
		//	{
		//		DW_LOG_FATAL("Failed to create Shaders");
		//		return false;
		//	}

		//	// Create general shader program
		//	dw::Shader* shaders[] = { m_prefilter_cs.get() };
		//	m_prefilter_program = std::make_unique<dw::Program>(1, shaders);

		//	if (!m_prefilter_program)
		//	{
		//		DW_LOG_FATAL("Failed to create Shader Program");
		//		return false;
		//	}
		//}

		{
			// Create general shaders
			m_sh_projection_cs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_COMPUTE_SHADER, "shader/sh_projection_cs.glsl"));
	
			if (!m_sh_projection_cs)
			{
				DW_LOG_FATAL("Failed to create Shaders");
				return false;
			}

			// Create general shader program
			dw::Shader* shaders[] = { m_sh_projection_cs.get() };
			m_sh_projection_program = std::make_unique<dw::Program>(1, shaders);

			if (!m_sh_projection_program)
			{
				DW_LOG_FATAL("Failed to create Shader Program");
				return false;
			}
		}

		{
			// Create general shaders
			m_sh_add_cs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_COMPUTE_SHADER, "shader/sh_add_cs.glsl"));
	
			if (!m_sh_add_cs)
			{
				DW_LOG_FATAL("Failed to create Shaders");
				return false;
			}

			// Create general shader program
			dw::Shader* shaders[] = { m_sh_add_cs.get() };
			m_sh_add_program = std::make_unique<dw::Program>(1, shaders);

			if (!m_sh_add_program)
			{
				DW_LOG_FATAL("Failed to create Shader Program");
				return false;
			}
		}

		{
			// Create general shaders
			m_mesh_vs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/mesh_vs.glsl"));
			m_mesh_fs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/mesh_fs.glsl"));

			if (!m_mesh_vs || !m_mesh_fs)
			{
				DW_LOG_FATAL("Failed to create Shaders");
				return false;
			}

			// Create general shader program
			dw::Shader* shaders[] = { m_mesh_vs.get(), m_mesh_fs.get() };
			m_mesh_program = std::make_unique<dw::Program>(2, shaders);

			if (!m_mesh_program)
			{
				DW_LOG_FATAL("Failed to create Shader Program");
				return false;
			}
		}

		{
			m_cubemap_vs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/sky_vs.glsl"));
			m_cubemap_fs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/sky_fs.glsl"));

			if (!m_cubemap_vs || !m_cubemap_fs)
			{
				DW_LOG_FATAL("Failed to create Shaders");
				return false;
			}

			// Create general shader program
			dw::Shader* shaders[] = { m_cubemap_vs.get(), m_cubemap_fs.get() };
			m_cubemap_program = std::make_unique<dw::Program>(2, shaders);

			if (!m_cubemap_program)
			{
				DW_LOG_FATAL("Failed to create Shader Program");
				return false;
			}
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_framebuffer()
	{
		m_env_cubemap = std::make_unique<dw::TextureCube>(ENVIRONMENT_MAP_SIZE, ENVIRONMENT_MAP_SIZE, 1, 1, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
		m_cubemap_depth = std::make_unique<dw::Texture2D>(ENVIRONMENT_MAP_SIZE, ENVIRONMENT_MAP_SIZE, 1, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
		
		m_env_cubemap->set_min_filter(GL_LINEAR);
		m_env_cubemap->set_mag_filter(GL_LINEAR);

		m_prefilter_cubemap = std::make_unique<dw::TextureCube>(PREFILTER_MAP_SIZE, PREFILTER_MAP_SIZE, 1, PREFILTER_MIP_LEVELS, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);

		m_sh_intermediate = std::make_unique<dw::Texture2D>(SH_INTERMEDIATE_SIZE * 9, SH_INTERMEDIATE_SIZE, 6, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);
	
		m_sh_intermediate->set_min_filter(GL_NEAREST);
		m_sh_intermediate->set_mag_filter(GL_NEAREST);
		
		m_sh = std::make_unique<dw::Texture2D>(9, 1, 1, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);

		m_sh->set_min_filter(GL_NEAREST);
		m_sh->set_mag_filter(GL_NEAREST);

		for (int i = 0; i < 6; i++)
		{
			m_cubemap_fbos.push_back(std::make_unique<dw::Framebuffer>());
			m_cubemap_fbos[i]->attach_render_target(0, m_env_cubemap.get(), i, 0, 0, true, true);
			m_cubemap_fbos[i]->attach_depth_stencil_target(m_cubemap_depth.get(), 0, 0);
		}
		
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool load_mesh()
	{
		m_mesh = dw::Mesh::load("mesh/teapot_smooth.obj");

		if (!m_mesh)
		{
			DW_LOG_FATAL("Failed to load mesh!");
			return false;
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void create_camera()
	{
		m_main_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 150.0f), glm::vec3(0.0f, 0.0, -1.0f));
        m_debug_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 150.0f), glm::vec3(0.0f, 0.0, -1.0f));
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void render_mesh(dw::Mesh* mesh)
	{
		// Bind vertex array.
		mesh->mesh_vertex_array()->bind();

		dw::SubMesh* submeshes = mesh->sub_meshes();

		for (uint32_t i = 0; i < mesh->sub_mesh_count(); i++)
		{
			dw::SubMesh& submesh = submeshes[i];
			// Issue draw call.
			glDrawElementsBaseVertex(GL_TRIANGLES, submesh.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * submesh.base_index), submesh.base_vertex);
		}
	}
    
	// -----------------------------------------------------------------------------------------------------------------------------------

	void render_meshes()
	{
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, m_width, m_height);

		glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
		glClearDepth(1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		// Bind shader program.
		m_mesh_program->use();

		glm::mat4 m = glm::mat4(1.0f);
		m_mesh_program->set_uniform("u_Model", glm::scale(m, glm::vec3(0.5f)));
		m_mesh_program->set_uniform("u_View", m_main_camera->m_view);
		m_mesh_program->set_uniform("u_Projection", m_main_camera->m_projection);

		// Draw meshes.
		render_mesh(m_mesh);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void render_cubemap()
	{
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_CULL_FACE);

		m_cubemap_program->use();
		m_cube_vao->bind();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, m_width, m_height);

		m_cubemap_program->set_uniform("u_Type", m_type);
		m_cubemap_program->set_uniform("u_View", m_main_camera->m_view);
		m_cubemap_program->set_uniform("u_Projection", m_main_camera->m_projection);

		if (m_cubemap_program->set_uniform("s_Cubemap", 0))
			m_env_cubemap->bind(0);

		if (m_cubemap_program->set_uniform("s_SH", 1))
			m_sh->bind(1);

		glDrawArrays(GL_TRIANGLES, 0, 36);

		glDepthFunc(GL_LESS);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void convert_env_map()
	{
		m_cubemap_convert_program->use();
		m_cube_vao->bind();

		for (int i = 0; i < 6; i++)
		{
			m_cubemap_fbos[i]->bind();

			glViewport(0, 0, ENVIRONMENT_MAP_SIZE, ENVIRONMENT_MAP_SIZE);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			m_cubemap_convert_program->set_uniform("u_Projection", m_capture_projection);
			m_cubemap_convert_program->set_uniform("u_View", m_capture_views[i]);

			if (m_cubemap_convert_program->set_uniform("s_EnvMap", 0))
				m_env_map->bind(0);
			
			glDrawArrays(GL_TRIANGLES, 0, 36);
		}

		m_env_cubemap->generate_mipmaps();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void project_cubemap_sh()
	{
		m_sh_projection_program->use();

		m_sh_projection_program->set_uniform("u_Width", (float)m_env_cubemap->width() / 4.0f);
		m_sh_projection_program->set_uniform("u_Height", (float)m_env_cubemap->height() / 4.0f);

		if (m_sh_projection_program->set_uniform("s_Cubemap", 1))
			m_env_cubemap->bind(1);

		m_sh_intermediate->bind_image(0, 0, 0, GL_WRITE_ONLY, GL_RGBA32F);

		glDispatchCompute(IRRADIANCE_CUBEMAP_SIZE / IRRADIANCE_WORK_GROUP_SIZE, IRRADIANCE_CUBEMAP_SIZE / IRRADIANCE_WORK_GROUP_SIZE, 6);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		m_sh_add_program->use();

		m_sh->bind_image(0, 0, 0, GL_WRITE_ONLY, GL_RGBA32F);

		if (m_sh_add_program->set_uniform("s_SHIntermediate", 1))
			m_sh_intermediate->bind(1);

		glDispatchCompute(9, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void prefilter_cubemap()
	{
		
	}

    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_camera()
    {
       dw::Camera* current = m_main_camera.get();
        
        if (m_debug_mode)
            current = m_debug_camera.get();
        
        float forward_delta = m_heading_speed * m_delta;
        float right_delta = m_sideways_speed * m_delta;
        
        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);

		m_camera_x = m_mouse_delta_x * m_camera_sensitivity;
		m_camera_y = m_mouse_delta_y * m_camera_sensitivity;
        
        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_camera_y),
                                                    (float)(m_camera_x),
                                                    (float)(0.0f)));
        }
        else
        {
            current->set_rotatation_delta(glm::vec3((float)(0),
                                                    (float)(0),
                                                    (float)(0)));
        }
        
        current->update();
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	void create_cube()
	{
		float vertices[] = 
		{
			// back face
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			 1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right       
			 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
			// front face
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			 1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
			 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			// left face
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			// right face
			 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			 1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			 1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			// bottom face
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			 1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
			 1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			 1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			-1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			// top face
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			 1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			 1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			 1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			-1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
		};

		m_cube_vbo = std::make_unique<dw::VertexBuffer>(GL_STATIC_DRAW, sizeof(vertices), vertices);

		if (!m_cube_vbo)
			DW_LOG_ERROR("Failed to create Vertex Buffer");

		// Declare vertex attributes.
		dw::VertexAttrib attribs[] =
		{
			{ 3, GL_FLOAT, false, 0 },
			{ 3, GL_FLOAT, false, (3 * sizeof(float)) },
			{ 2, GL_FLOAT, false, (6 * sizeof(float)) }
		};

		// Create vertex array.
		m_cube_vao = std::make_unique<dw::VertexArray>(m_cube_vbo.get(), nullptr, (8 * sizeof(float)), 3, attribs);

		m_capture_projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
		m_capture_views =
		{
			glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
			glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
			glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
			glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
			glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
			glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
		};
	}
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
private:
	// General GPU resources.
	std::vector<std::unique_ptr<dw::Framebuffer>> m_cubemap_fbos;
	std::vector<glm::mat4> m_capture_views;
	glm::mat4 m_capture_projection;

	std::unique_ptr<dw::VertexBuffer> m_cube_vbo;
	std::unique_ptr<dw::VertexArray> m_cube_vao;

	std::unique_ptr<dw::UniformBuffer> m_object_ubo;
    std::unique_ptr<dw::UniformBuffer> m_global_ubo;

	std::unique_ptr<dw::Texture2D> m_cubemap_depth;
	std::unique_ptr<dw::Texture2D> m_env_map;
	std::unique_ptr<dw::TextureCube> m_env_cubemap;
	std::unique_ptr<dw::TextureCube> m_prefilter_cubemap;
	std::unique_ptr<dw::Texture2D> m_sh;
	std::unique_ptr<dw::Texture2D> m_sh_intermediate;

	std::unique_ptr<dw::Shader>  m_cubemap_convert_vs;
	std::unique_ptr<dw::Shader>  m_cubemap_convert_fs;
	std::unique_ptr<dw::Program> m_cubemap_convert_program;

	std::unique_ptr<dw::Shader>  m_cubemap_vs;
	std::unique_ptr<dw::Shader>  m_cubemap_fs;
	std::unique_ptr<dw::Program> m_cubemap_program;

	std::unique_ptr<dw::Shader>  m_mesh_vs;
	std::unique_ptr<dw::Shader>  m_mesh_fs;
	std::unique_ptr<dw::Program> m_mesh_program;

	std::unique_ptr<dw::Shader>  m_sh_projection_cs;
	std::unique_ptr<dw::Program> m_sh_projection_program;

	std::unique_ptr<dw::Shader>  m_sh_add_cs;
	std::unique_ptr<dw::Program> m_sh_add_program;

	std::unique_ptr<dw::Shader>  m_prefilter_cs;
	std::unique_ptr<dw::Program> m_prefilter_program;

    // Camera.
	std::unique_ptr<dw::Camera> m_main_camera;
    std::unique_ptr<dw::Camera> m_debug_camera;

	// Mesh
	dw::Mesh* m_mesh;

    // Camera controls.
	bool m_show_gui = true;
    bool m_mouse_look = false;
    bool m_debug_mode = false;
    float m_heading_speed = 0.0f;
    float m_sideways_speed = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed = 0.06f;
	float m_camera_x = 0.0f;
	float m_camera_y = 0.0f;
	int m_type = 0;
};

DW_DECLARE_MAIN(RuntimeIBL)
