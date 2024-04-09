use std::io::Read;
use std::default::Default;
use ash;
use ash::vk;

pub unsafe fn load_shader_module(
    file_path: String,
    device: &ash::Device,
) -> ash::prelude::VkResult<vk::ShaderModule>{
    let mut file = std::fs::File::open(file_path)
        .expect("Unable to open file!");

    let mut file_data: Vec<u8> = vec![];
    file.read_to_end(&mut file_data)
        .expect("Unable to read file!");


    let mut create_info = vk::ShaderModuleCreateInfo::builder();
    create_info.p_code = file_data.as_ptr().cast();
    create_info.code_size = (file_data.len());

    device.create_shader_module(&create_info, None)
}

pub struct PipelineBuilder{
    pub shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    pub input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterizer: vk::PipelineRasterizationStateCreateInfo,
    pub color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    pub multisampling: vk::PipelineMultisampleStateCreateInfo,
    pub pipeline_layout: vk::PipelineLayout,
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfo,
    pub render_info: vk::PipelineRenderingCreateInfo,
    pub color_attachment_format: vk::Format,
}

impl PipelineBuilder{
    pub fn new() -> Self{
        Self{
            shader_stages: vec![],
            input_assembly: Default::default(),
            rasterizer: Default::default(),
            color_blend_attachment: Default::default(),
            multisampling: Default::default(),
            pipeline_layout: Default::default(),
            depth_stencil: Default::default(),
            render_info: Default::default(),
            color_attachment_format: Default::default(),
        }
    }

    pub fn clear(&mut self) {
        self.input_assembly = Default::default();
        self.rasterizer = Default::default();
        self.color_blend_attachment = Default::default();
        self.multisampling = Default::default();
        self.pipeline_layout = Default::default();
        self.depth_stencil = Default::default();
        self.render_info = Default::default();
        self.shader_stages.clear();
    }

    pub unsafe fn build_pipeline(&mut self, device: &ash::Device)
    -> vk::Pipeline{
        let viewport_state =
            vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let attachments = [self.color_blend_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments);

        let vertex_input_info =
            vk::PipelineVertexInputStateCreateInfo::default();

        let states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&states);

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(self.shader_stages.as_slice())
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.pipeline_layout)
            .dynamic_state(&dynamic_info);

        pipeline_info.p_next = &self.render_info as *const _ as *mut _;


        let create_infos = [pipeline_info.build()];

        *device.create_graphics_pipelines(
            vk::PipelineCache::null(), &create_infos, None

        ).expect("Unable to build pipeline!")
            .first()
            .expect("Couldn't get single pipeline!")
    }

    pub fn set_shaders(&mut self, vertex: vk::ShaderModule, fragment: vk::ShaderModule){
        self.shader_stages.clear();

        self.shader_stages.push(
            super::vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::VERTEX, vertex
            ).build()
        );

        self.shader_stages.push(
            super::vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT, fragment
            ).build()
        );
    }

    pub fn set_input_topology(&mut self, topology: vk::PrimitiveTopology){
        self.input_assembly.topology = topology;

        self.input_assembly.primitive_restart_enable = vk::FALSE;
    }

    pub fn set_polygon_mode(&mut self, mode: vk::PolygonMode){
        self.rasterizer.polygon_mode = mode;
        self.rasterizer.line_width = 1.0;
    }

    pub fn set_cull_mode(&mut self, cull: vk::CullModeFlags, front_face: vk::FrontFace){
        self.rasterizer.cull_mode = cull;
        self.rasterizer.front_face = front_face;
    }

    pub fn set_multisampling_none(&mut self){
        self.multisampling.sample_shading_enable = vk::FALSE;

        self.multisampling.rasterization_samples = vk::SampleCountFlags::TYPE_1;

        self.multisampling.min_sample_shading = 1.0;

        self.multisampling.alpha_to_coverage_enable = vk::FALSE;

        self.multisampling.alpha_to_one_enable = vk::FALSE;
    }

    pub fn disable_blending(&mut self){
        self.color_blend_attachment.color_write_mask =
                vk::ColorComponentFlags::R |
                vk::ColorComponentFlags::G |
                vk::ColorComponentFlags::B |
                vk::ColorComponentFlags::A;

        self.color_blend_attachment.blend_enable = vk::FALSE;
    }

    pub fn set_color_attachment_format(&mut self, format: vk::Format){
        self.color_attachment_format = format;

        self.render_info.color_attachment_count = 1;
        self.render_info.p_color_attachment_formats = &format as *const _;
    }

    pub fn set_depth_format(&mut self, format: vk::Format){
        self.render_info.depth_attachment_format = format;
    }

    pub fn disable_depthtest(&mut self){
        self.depth_stencil.depth_test_enable = vk::FALSE;
        self.depth_stencil.depth_write_enable = vk::FALSE;
        self.depth_stencil.depth_compare_op = vk::CompareOp::NEVER;
        self.depth_stencil.depth_bounds_test_enable = vk::FALSE;
        self.depth_stencil.stencil_test_enable = vk::FALSE;
        self.depth_stencil.front = vk::StencilOpState::default();
        self.depth_stencil.back = vk::StencilOpState::default();
        self.depth_stencil.min_depth_bounds = 0.0;
        self.depth_stencil.max_depth_bounds = 1.0;
    }
}