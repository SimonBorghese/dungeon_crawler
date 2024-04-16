#![allow(dead_code)]

use std::collections::VecDeque;
use std::ops::BitOr;
use sdl2::event::WindowEvent;
use ash_bootstrap;
use ash;
use ash::vk;
use ash::vk::{CommandBufferResetFlags, CommandPoolCreateFlags, Handle};
use ash_bootstrap::QueueFamilyCriteria;
use super::vk_loader::*;
use super::vk_initializers::*;
use super::vk_image;
use super::vk_types;
use super::vk_descriptor::*;
use super::vk_pipelines::*;
use super::vk_image::*;
use super::vk_loader::*;
use vk_mem;
use vk_mem::Alloc;
use imgui;
use imgui_rs_vulkan_renderer;
use imgui_sdl2;
use num;
use crate::engine::vk_types::AllocatedBuffer;

const USE_VALIDATION_LAYERS: bool = true;

#[derive(Default)]
pub struct DeletionQueue{
    pub deletors: VecDeque<Box<dyn Fn(&VulkanEngine)>>,
    pub image_views: VecDeque<vk::ImageView>,
    pub images: VecDeque<vk::Image>,
    pub allocations: VecDeque<vk_mem::Allocation>,
    pub descriptors: VecDeque<DescriptorAllocator>,
    pub pipeline_layouts: VecDeque<vk::PipelineLayout>,
    pub pipelines: VecDeque<vk::Pipeline>
}

impl DeletionQueue{
    pub fn new() -> Self{
        Self{
            deletors: VecDeque::new(),
            image_views: VecDeque::new(),
            images: VecDeque::new(),
            allocations: VecDeque::new(),
            descriptors: VecDeque::new(),
            pipeline_layouts: VecDeque::new(),
            pipelines: VecDeque::new(),
        }
    }
    pub fn push_function(&mut self, func: Box<dyn Fn(&VulkanEngine)>){
        self.deletors.push_front(func);
    }

    pub fn push_image_view(&mut self, view: vk::ImageView){
        self.image_views.push_front(view);
    }

    pub fn push_image(&mut self, view: vk::Image){
        self.images.push_front(view);
    }

    pub fn push_alloc(&mut self, alloc: vk_mem::Allocation){
        self.allocations.push_front(alloc);
    }

    pub fn push_descriptor(&mut self, descriptor: DescriptorAllocator){
        self.descriptors.push_front(descriptor);
    }

    pub fn push_pipeline_layout(&mut self, layout: vk::PipelineLayout){
        self.pipeline_layouts.push_front(layout);
    }

    pub fn push_pipeline(&mut self, pipeline: vk::Pipeline){
        self.pipelines.push_front(pipeline);
    }

    pub unsafe fn flush(&self, engine: &VulkanEngine){

        for func in self.deletors.iter().enumerate(){
            func.1(engine)
        }

        for view in self.image_views.iter().enumerate(){
            engine.get_device().destroy_image_view(
                *view.1, None
            );
        }

        for image in self.images.iter().enumerate(){
            engine.get_device().destroy_image(
                *image.1, None
            );
        }

        for alloc in self.allocations.iter().enumerate(){
            engine.allocator.as_ref().unwrap().free_memory(
                alloc.1
            );
        }

        for desc in self.descriptors.iter().enumerate(){
            desc.1.clear_descriptors(engine.get_device());
            desc.1.destroy_pool(engine.get_device());
        }

        for layout in self.pipeline_layouts.iter().enumerate(){
            engine.get_device().destroy_pipeline_layout(
                *layout.1, None
            );
        }

        for pipeline in self.pipelines.iter().enumerate(){
            engine.get_device().destroy_pipeline(
                *pipeline.1, None
            );
        }
    }

    pub fn clear(&mut self){
        self.deletors.clear();
    }
}

#[derive(Default)]
pub struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    render_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
    deletion_queue: DeletionQueue,
}

const FRAME_OVERLAP: usize = 2;

pub struct VulkanEngine{
    pub sdl: sdl2::Sdl,
    pub video: sdl2::VideoSubsystem,
    pub window: sdl2::video::Window,
    pub event: sdl2::EventPump,
    pub entry: ash::Entry,
    pub is_initialized: bool,
    pub frame_number: i32,
    pub stop_rendering: bool,
    pub window_extent: vk::Extent2D,

    pub instance: Option<ash::Instance>,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub chosen_gpu: vk::PhysicalDevice,
    pub device: Option<ash::Device>,
    pub surface: vk::SurfaceKHR,
    pub surface_dev: Option<ash::extensions::khr::Surface>,
    pub debug_utils: Option<ash::extensions::ext::DebugUtils>,
    
    pub swapchain: Option<ash_bootstrap::swapchain::Swapchain>,
    pub swapchain_dev: Option<ash::extensions::khr::Swapchain>,
    pub swapchain_image_format: vk::Format,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_extent: vk::Extent2D,
    
    pub frames: Vec<FrameData>,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,

    pub allocator: Option<vk_mem::Allocator>,

    pub draw_image: vk_types::AllocatedImage,
    
    pub depth_image: vk_types::AllocatedImage,
    
    pub draw_extent: vk::Extent2D,

    pub global_descriptor_allocator: DescriptorAllocator,

    pub draw_image_descriptors: vk::DescriptorSet,

    pub draw_image_description_layout: vk::DescriptorSetLayout,

    pub imm_fence: vk::Fence,

    pub imm_command_buffer: vk::CommandBuffer,

    pub imm_command_pool: vk::CommandPool,

    pub im_context: Option<imgui::Context>,

    pub im_sdl2: Option<imgui_sdl2::ImguiSdl2>,

    pub im_render: Option<imgui_rs_vulkan_renderer::Renderer>,

    pub triangle_pipeline_layout: vk::PipelineLayout,

    pub triangle_pipeline: vk::Pipeline,

    pub main_deletion_queue: DeletionQueue,

    pub test_mesh: MeshAsset,
}

impl VulkanEngine{
    pub fn get_current_frame(&self) -> usize{
        self.frame_number as usize % FRAME_OVERLAP
    }

    #[inline]
    fn get_device(&self) -> &ash::Device{
        self.device.as_ref()
            .expect("Unable to get device!")
    }
    
    pub fn new(width: u32, height: u32) -> Self {
        let sdl = sdl2::init()
            .expect("Unable to initialize SDL2");
        let video = sdl.video()
            .expect("Unable to get SDL video!");
        let event = sdl.event_pump()
            .expect("Unable to get event pump!");
        let entry: ash::Entry;
        unsafe {
            entry = ash::Entry::load()
                .expect("Unable to load vulkan lib!");
        }
        let window = video.window("Vulkan Engine", width, height)
            .vulkan()
            .position_centered()
            .build()
            .expect("Unable to build Window");

        Self {
            sdl,
            video,
            window,
            event,
            entry,
            is_initialized: false,
            frame_number: 0,
            stop_rendering: false,
            window_extent: vk::Extent2D::builder().width(width).height(height).build(),
            instance: None,
            debug_messenger: Default::default(),
            debug_utils: None,
            chosen_gpu: std::default::Default::default(),
            device: std::default::Default::default(),
            surface: Default::default(),
            surface_dev: None,
            swapchain: None,
            swapchain_dev: None,
            swapchain_image_format: Default::default(),
            swapchain_images: vec![],
            swapchain_image_views: vec![],
            swapchain_extent: Default::default(),
            frames: {
                let mut frames: Vec<FrameData> = vec![];
                for _ in 0..FRAME_OVERLAP{
                    frames.push(FrameData::default());
                }
                frames
            },
            graphics_queue: Default::default(),
            graphics_queue_family: 0,
            allocator: None,
            draw_image: Default::default(),
            depth_image: Default::default(),
            draw_extent: Default::default(),
            global_descriptor_allocator: DescriptorAllocator::new(),
            draw_image_descriptors: Default::default(),
            draw_image_description_layout: Default::default(),
            imm_fence: Default::default(),
            imm_command_buffer: Default::default(),
            imm_command_pool: Default::default(),
            im_context: None,
            im_sdl2: None,
            im_render: None,
            triangle_pipeline_layout: Default::default(),
            main_deletion_queue: Default::default(),
            triangle_pipeline: Default::default(),
            test_mesh: Default::default(),
        }
    }

    pub fn init(&mut self){

        unsafe {
            self.init_vulkan()
                .expect("Unable to initialize Vulkan");
            self.init_swapchain();
            self.init_commands();
            self.init_sync_structures();
            self.init_descriptors();
            self.init_pipelines();
            self.init_imgui();
            self.init_default_data();
        }

        self.is_initialized = true;
    }

    pub fn run(&mut self){
        let mut quit = false;

        while !quit{
            for event in self.event.poll_iter(){
                match event{
                    sdl2::event::Event::Quit {..} => {
                        quit = true
                    }

                    sdl2::event::Event::Window {win_event, ..} => {
                        match win_event{
                            WindowEvent::Minimized => {
                                self.stop_rendering = true;
                            }
                            WindowEvent::Restored => {
                                self.stop_rendering = false;
                            }
                            _ => {}
                        }
                    }

                    _ => {}
                }
            }

            if self.stop_rendering{
                continue;
            }

            unsafe{
                self.draw();
            }
        }
    }

    pub fn immediate_submit(&mut self, func: &dyn Fn(
        &ash::Device,
        vk::CommandBuffer
    )){
        unsafe {
            let fences = [self.imm_fence];
            self.get_device().reset_fences(&fences)
                .expect("Unable to reset fences!");

            self.get_device().reset_command_buffer(self.imm_command_buffer,
            CommandBufferResetFlags::empty())
                .expect("Unable to reset command buffer!");

            //let cmd = self.imm_command_buffer;
            let cmd_begin_info = command_buffer_begin_info(
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            );

            self.get_device().begin_command_buffer(self.imm_command_buffer, &cmd_begin_info)
                .expect("Unable to begin command buffer!");

           func(self.get_device(), self.imm_command_buffer);

            self.get_device().end_command_buffer(self.imm_command_buffer)
                .expect("Unable to end command buffer!");

            let cmd_info = command_buffer_submit_info(self.imm_command_buffer).build();
            let cmd_infos = [cmd_info];
            let submit = submit_info(&cmd_infos,
                                     &[],
                                     &[])
                .build();

            let submits=  [submit];
            self.get_device().queue_submit2(self.graphics_queue, &submits, self.imm_fence)
                .expect("Unable to submit to queue!");

            self.get_device().wait_for_fences(&fences, true, u64::MAX)
                .expect("Unable to wait for fence!");
        }
    }

    unsafe fn draw(&mut self){
        let current_frame = &self.frames[self.get_current_frame()];
        self.get_device().wait_for_fences(&[current_frame.render_fence],
                                          true, 1000000000)
            .expect("Unable to wait for fence");

        current_frame.deletion_queue.flush(self);

        self.get_device().reset_fences(&[current_frame.render_fence])
            .expect("Unable to reset fence!");

        let swapchain_image = self.swapchain.as_mut().unwrap()
            .acquire(self.device.as_mut().unwrap(), self.surface_dev.as_ref().unwrap(),
                     1000000000, false)
            .expect("Unable to acquire swapchain image");

        let cmd = current_frame.command_buffer;

        self.get_device().reset_command_buffer(cmd, Default::default())
            .expect("Unable to reset command buffer!");

        let cmd_begin_info = command_buffer_begin_info(
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        );

        self.draw_extent.width = self.draw_image.image_extent.width;
        self.draw_extent.height = self.draw_image.image_extent.height;

        self.get_device().begin_command_buffer(cmd, &cmd_begin_info)
            .expect("Unable to begin command buffer!");

        transition_image(self.get_device(),
        cmd, self.draw_image.image,
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        transition_image(self.get_device(),
                         cmd, self.draw_image.image,
                         vk::ImageLayout::GENERAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        transition_image(self.get_device(),
                         cmd, self.depth_image.image,
                         vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);

        self.draw_geometry(cmd);

        transition_image(self.get_device(),
                                   cmd, self.draw_image.image,
                                   vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        transition_image(self.get_device(),
        cmd, self.swapchain_images[swapchain_image.image_index],
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        copy_image_to_image(self.get_device(), cmd,
                            self.draw_image.image,
        self.swapchain_images[swapchain_image.image_index],
            self.draw_extent, self.swapchain_extent);

        transition_image(self.get_device(),
                                   cmd, self.swapchain_images[swapchain_image.image_index],
                                   vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                   vk::ImageLayout::PRESENT_SRC_KHR);

        self.get_device().end_command_buffer(cmd)
            .expect("Unable to end command buffer!");



        let cmd_info = command_buffer_submit_info(cmd)
            .build();

        let wait_info = semaphore_submit_info(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            swapchain_image.ready
        ).build();

        let signal_info = semaphore_submit_info(
            vk::PipelineStageFlags2::ALL_GRAPHICS,
            current_frame.render_semaphore
        ).build();

        let submit = submit_info(&[cmd_info],
                                 &[signal_info],
                                 &[wait_info])
            .build();

        self.get_device().queue_submit2(self.graphics_queue, &[submit],
                                        current_frame.render_fence)
            .expect("Unable to submit queue");


        let semaphore = current_frame.render_semaphore;
        self.swapchain.as_mut().unwrap().queue_present(self.graphics_queue,
                                                       semaphore, swapchain_image.image_index)
            .expect("Unable to present image!");

        self.get_device().reset_fences(&[swapchain_image.complete])
            .expect("Unable to reset fence!");
        self.frame_number += 1;
        println!("Frame Number: {}", self.frame_number);
    }

    unsafe fn init_default_data(&mut self){

        self.test_mesh = MeshAsset::load_first_mesh(self, std::path::PathBuf::from(
            "assets/test.glb"
        ))
            .expect("Unable to load test mesh");
    }

    pub unsafe fn draw_geometry(&self, cmd: vk::CommandBuffer){
        let color_attachment = attachment_info(
            self.draw_image.image_view, None, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        ).build();

        let depth_attachment = depth_attachment_info(
            self.depth_image.image_view, None, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        ).build();

        let color_attachments = [color_attachment];

        let render_info = vk::RenderingInfo::builder()
            .render_area(vk::Rect2D::builder()
                .extent(self.draw_extent)
                .offset(vk::Offset2D::default())
                .build())
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment)
            .layer_count(1);

        self.get_device().cmd_begin_rendering(cmd, &render_info);

        self.get_device().cmd_bind_pipeline(cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.triangle_pipeline);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.draw_extent.width as f32)
            .height(self.draw_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        let viewports = [viewport];
        self.get_device().cmd_set_viewport(cmd, 0, &viewports);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D::builder()
                .x(0)
                .y(0).build())
            .extent(self.draw_extent)
            .build();

        let scissors = [scissor];

        self.get_device().cmd_set_scissor(cmd, 0, &scissors);

        let mut world_matrix: glm::Mat4 = num::one();

        world_matrix = world_matrix.mul_m(&glm::ext::perspective(
            glm::radians(45.0),
            self.window_extent.width as f32 / self.window_extent.height as f32,
            0.1, 1000.0
        ));

        world_matrix = world_matrix.mul_m(
            &glm::ext::translate(
                &num::one(), glm::vec3(0.0,  0.0, 5.0 * glm::sin(self.frame_number as f32 / 120.0))
            )
        );

        world_matrix = world_matrix.mul_m(
            &glm::ext::rotate(
                &num::one(), glm::radians(180.0),
                glm::vec3(1.0, 0.0, 0.0)
            )
        );

        world_matrix = world_matrix.mul_m(
            &glm::ext::rotate(
                &num::one(), glm::radians(360.0) * glm::sin(self.frame_number as f32 / 30.0),
                glm::vec3(0.0, 1.0, 0.0)
            )
        );


        let push_constants= vk_types::GPUDrawPushConstants{
            world_matrix,
            vertex_buffer: self.test_mesh.mesh_buffers.vertex_buffer_address,
        };

        self.get_device().cmd_push_constants(cmd, self.triangle_pipeline_layout,
        vk::ShaderStageFlags::VERTEX, 0, std::slice::from_raw_parts(
                &push_constants as *const _ as *const u8,
                std::mem::size_of::<vk_types::GPUDrawPushConstants>()
            ));

        self.get_device().cmd_bind_index_buffer(
            cmd, self.test_mesh.mesh_buffers.index_buffer.buffer,
            vk::DeviceSize::from(0u32), vk::IndexType::UINT32
        );

        for surface in &self.test_mesh.surfaces {
            self.get_device().cmd_draw_indexed(cmd, surface.count, 1,
                                               surface.start_index, 0, 0);
        }

        self.get_device().cmd_end_rendering(cmd);

    }


    extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        type_: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let data = unsafe { *data };
        let message = unsafe { std::ffi::CStr::from_ptr(data.p_message) }.to_string_lossy();

        println!("================VULKAN ERROR===================");
        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            println!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            println!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            println!("({:?}) {}", type_, message);
        } else {
            println!("({:?}) {}", type_, message);
        }
        println!("-------------------------------------------------\n\n\n");

        vk::FALSE
    }
    unsafe fn init_vulkan(&mut self) -> Result<(), ash_bootstrap::InstanceCreationError>{
        let callback: vk::PFN_vkDebugUtilsMessengerCallbackEXT = Some(Self::debug_callback);
        let builder = ash_bootstrap::InstanceBuilder::new()
            .app_name("Vulkan Application").unwrap()
            .validation_layers({
                match USE_VALIDATION_LAYERS {
                    true => ash_bootstrap::ValidationLayers::Require,
                    false => ash_bootstrap::ValidationLayers::Disable
                }
            })
            .request_debug_messenger(
                ash_bootstrap::DebugMessenger::Custom {callback, user_data_pointer: 0 as _}
                )
            .require_api_version(1, 3)
            .require_surface_extensions(&self.window).expect("Unable to request extensions")
            .build(&self.entry)?;

        let instance = builder.0;
        self.instance = Some(instance);
        self.debug_utils = Some(builder.1.0);
        self.debug_messenger = builder.1.1.unwrap();

        let surface_handle = self.window.vulkan_create_surface(
                self.instance.as_ref().unwrap().handle().as_raw() as _)
            .expect("Unable to make Vulkan surface");

        self.surface = vk::SurfaceKHR::from_raw(surface_handle);
        self.surface_dev =
            Some(ash::extensions::khr::Surface::new(&self.entry, self.instance.as_ref().unwrap()));


        let vulkan_11_features = vk::PhysicalDeviceVulkan11Features::builder()
            .build();


        let vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true)
            .build();

        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .build();

        //vulkan_11_features.p_next = &shader_objects as *const _ as *mut _;
        vulkan_12_features.p_next = &vulkan_13_features as *const _ as *mut _;

        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .build();
        features.p_next = &vulkan_12_features as *const _ as *mut _;

        let selector = ash_bootstrap::DeviceBuilder::new()
            .require_version(1,3)
            .set_required_features_13(vulkan_13_features)
            .set_required_features_12(vulkan_12_features)
            .set_required_features_11(vulkan_11_features)
            .for_surface(self.surface)
            .require_extension(ash::extensions::khr::Swapchain::name().as_ptr())
            .require_extension(ash::extensions::ext::ShaderObject::name().as_ptr())
            .queue_family(QueueFamilyCriteria::graphics_present())
            .build(self.instance.as_ref().unwrap(),
            self.surface_dev.as_ref().unwrap(), &builder.2)
            .expect("Unable to build device!");

        self.device = Some(selector.0);
        self.chosen_gpu = selector.1.physical_device();

        println!("Successfully made device: {}", selector.1.device_name());


        self.graphics_queue_family = Self::get_graphics_queue_family(&selector.1)
            .expect("Unable to get graphics queue family!") as u32;

        self.graphics_queue = self.get_device()
            .get_device_queue(self.graphics_queue_family, 0);

        let allocator_info = vk_mem::AllocatorCreateInfo::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap(),
            self.chosen_gpu
        )
            .flags(vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS);

        self.allocator = Some(vk_mem::Allocator::new(allocator_info)
            .expect("Unable to create the allocator!"));

        Ok(())
    }

    unsafe fn get_graphics_queue_family(device: &ash_bootstrap::DeviceMetadata) -> Result<usize, &str>{
        for queue_family in device.queue_family_properties()
            .iter()
            .enumerate(){
            if queue_family.1.queue_flags.as_raw() & vk::QueueFlags::GRAPHICS.as_raw() > 0{
                return Ok(queue_family.0);
            }
        }
        return Err("No Graphics Queue Family Found")
    }


    unsafe fn init_swapchain(&mut self){
        self.create_swapchain(800,600);

    }

    unsafe fn create_swapchain(&mut self, width: u32, height: u32){
        self.swapchain_image_format = vk::Format::B8G8R8A8_UNORM;
        self.swapchain_extent = vk::Extent2D::builder()
            .width(width)
            .height(height)
            .build();

        let swapchain_builder = ash_bootstrap::SwapchainOptions::new()
            .format_preference(&[vk::SurfaceFormatKHR::builder()
                                   .format(self.swapchain_image_format)
                                   .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                                    .build()])
            .frames_in_flight(3)
            .present_mode_preference(&[vk::PresentModeKHR::FIFO])
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT
             | vk::ImageUsageFlags::TRANSFER_DST);

        let mut swapchain_builder2 =
            ash_bootstrap::Swapchain::new(swapchain_builder.clone(), self.surface, self.chosen_gpu,
            self.get_device(), ash::extensions::khr::Swapchain::new(self.instance.as_ref().unwrap(),
                                          self.get_device()), self.swapchain_extent);


        swapchain_builder2.acquire(self.get_device(),
                                   self.surface_dev.as_ref().unwrap(),
                                   1000000000, false)
            .expect("Unable to acquire images");
        self.swapchain = Some(swapchain_builder2);
        self.swapchain_dev = Some(ash::extensions::khr::Swapchain::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap()
        ));
        self.swapchain_images = self.swapchain.as_ref().unwrap().images().to_vec();
        self.swapchain_image_views = self.create_image_views(&self.swapchain_images);

        let draw_image_extent = vk::Extent3D::builder()
            .width(width)
            .height(height)
            .depth(1)
            .build();

        self.draw_image.image_format = vk::Format::R16G16B16A16_SFLOAT;
        self.draw_image.image_extent = draw_image_extent;

        let mut draw_image_usages = vk::ImageUsageFlags::empty();
        draw_image_usages |= vk::ImageUsageFlags::TRANSFER_SRC;
        draw_image_usages |= vk::ImageUsageFlags::TRANSFER_DST;
        draw_image_usages |= vk::ImageUsageFlags::STORAGE;
        draw_image_usages |= vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let rimg_info = image_create_info(
            self.draw_image.image_format,
            draw_image_usages,
            self.draw_image.image_extent);

        let mut rimg_allocinfo = vk_mem::AllocationCreateInfo::default();
        rimg_allocinfo.usage = vk_mem::MemoryUsage::AutoPreferDevice;
        rimg_allocinfo.required_flags = {
            let mut flags = vk::MemoryPropertyFlags::default();
            flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
            flags
        };

        let image_output =
            self.allocator.as_ref().unwrap().create_image(&rimg_info, &rimg_allocinfo)
            .expect("Unable to create image!");
        self.draw_image.image = image_output.0;
        self.draw_image.allocation = Some(image_output.1);

        let image_view_info = imageview_create_info(
            self.draw_image.image_format,
            self.draw_image.image,
            vk::ImageAspectFlags::COLOR
        );

        self.draw_image.image_view = self.get_device().create_image_view(&image_view_info, None)
            .expect("Unable to create image view!");


        self.depth_image.image_format = vk::Format::D32_SFLOAT;

        let depth_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;

        let dimg_info = image_create_info(
            self.depth_image.image_format, depth_usage, self.draw_image.image_extent
        );

        let depth_output = self.allocator.as_ref().unwrap()
            .create_image(&dimg_info, &rimg_allocinfo)
            .expect("Unable to create depth image!");

        self.depth_image.image = depth_output.0;
        self.depth_image.allocation = Some(depth_output.1);

        let dview_info = imageview_create_info(
            self.depth_image.image_format, self.depth_image.image,
            vk::ImageAspectFlags::DEPTH
        );

        self.depth_image.image_view = self.get_device().create_image_view(
            &dview_info, None
        ).expect("Unable to create depth image view!");

        self.main_deletion_queue.push_image_view(self.depth_image.image_view);
        self.main_deletion_queue.push_image(self.depth_image.image);
        self.main_deletion_queue.push_alloc(self.depth_image.allocation.unwrap());
        self.main_deletion_queue.push_image_view(self.draw_image.image_view);
        self.main_deletion_queue.push_image(self.draw_image.image);
        self.main_deletion_queue.push_alloc(self.draw_image.allocation.unwrap());


    }

    unsafe fn create_image_views(&self, images: &Vec<vk::Image>)
        -> Vec<vk::ImageView>{
        let components = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY)
            .build();

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let mut image_views: Vec<vk::ImageView> = vec![];
        for image in images{
            let info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.swapchain_image_format)
                .components(components)
                .subresource_range(subresource_range);

            image_views.push(self.get_device().create_image_view(&info, None)
                .expect("Unable to make image view"));
        }
        image_views
    }

    unsafe fn destroy_swapchain(&mut self){
        self.swapchain.as_mut().unwrap().destroy(self.device.as_ref().unwrap());

        for image in &self.swapchain_image_views{
            self.get_device().destroy_image_view(*image, None);
        }
    }

    unsafe fn init_commands(&mut self){
        let command_pool_info = command_pool_create_info(
            self.graphics_queue_family,
            CommandPoolCreateFlags::RESET_COMMAND_BUFFER
        );

        for i in 0..FRAME_OVERLAP{
            self.frames[i].command_pool = self.get_device().create_command_pool(&command_pool_info, None)
                .expect("Unable to build command pool");

            let command_buffer_info = command_buffer_allocate_info(
                self.frames[i].command_pool,
                1
            );

            self.frames[i].command_buffer = *self.get_device().allocate_command_buffers(&command_buffer_info)
                .expect("Unable to create command buffer").first()
                .expect("No Command Buffers Found");
        }

        self.imm_command_pool = self.get_device().create_command_pool(&command_pool_info, None)
            .expect("Unable to create command pool!");

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.imm_command_pool)
            .command_buffer_count(1);

        self.imm_command_buffer = *self.get_device().allocate_command_buffers(&cmd_alloc_info)
            .expect("Unable to allocate command buffer!")
            .first().expect("Unable to get single command buffer");

        self.main_deletion_queue.push_function(Box::new(|device: &VulkanEngine|{
            device.get_device().destroy_command_pool(device.imm_command_pool, None);
        }));
    }

    unsafe fn init_sync_structures(&mut self){
        let fence_info = fence_create_info(vk::FenceCreateFlags::SIGNALED);
        let semaphore_info = semaphore_create_info();

        for i in 0..FRAME_OVERLAP{
            self.frames[i].render_fence = self.get_device().create_fence(&fence_info, None)
                .expect("Unable to create fence!");

            self.frames[i].render_semaphore = self.get_device()
                .create_semaphore(&semaphore_info, None)
                .expect("Unable to create semaphore!");
        }

        self.imm_fence = self.get_device().create_fence(&fence_info, None)
            .expect("Unable to create fence!");

        self.main_deletion_queue.push_function(Box::new(|device: &VulkanEngine|{
            device.get_device().destroy_fence(device.imm_fence, None);
        }));
    }

    unsafe fn init_descriptors(&mut self){
        let sizes: Vec<PoolSizeRatio> = vec![
            PoolSizeRatio{
                desc_type: vk::DescriptorType::STORAGE_IMAGE,
                ratio: 1.0,
            }
        ];

        self.global_descriptor_allocator.init_pool(self.device.as_ref().unwrap(),
                                                   10, sizes);

        let mut builder = DescriptorLayoutBuilder::new();
        builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);
        self.draw_image_description_layout = builder.build(self.get_device(),
                                                           vk::ShaderStageFlags::COMPUTE);

        self.draw_image_descriptors = *self.global_descriptor_allocator
            .allocate(self.device.as_ref().unwrap(), self.draw_image_description_layout)
            .first()
            .expect("Unable to get a single descriptor!");

        let img_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(self.draw_image.image_view)
            .build();

        let image_infos = [img_info];

        let draw_image_write = vk::WriteDescriptorSet::builder()
            .dst_binding(0)
            .dst_set(self.draw_image_descriptors)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos)
            .build();

        let descriptor_writes = [draw_image_write];
        self.get_device().update_descriptor_sets(&descriptor_writes,
        &[]);

        self.main_deletion_queue.push_descriptor(self.global_descriptor_allocator.clone());
    }

    unsafe fn init_pipelines(&mut self){
        self.init_triangle_pipeline();
    }

    unsafe fn init_imgui(&mut self){
        let mut imgui = imgui::Context::create();
        let platform = imgui_sdl2::ImguiSdl2::new(&mut imgui, &self.window);
        let renderer = imgui_rs_vulkan_renderer::
        Renderer::with_default_allocator(
            self.instance.as_ref().unwrap(),
            self.chosen_gpu,
            self.device.clone().unwrap(),
            self.graphics_queue,
            self.imm_command_pool,
            imgui_rs_vulkan_renderer::DynamicRendering {
                color_attachment_format: self.swapchain_image_format,
                depth_attachment_format: None,
            },
            &mut imgui,
            None
        )
            .expect("Unable to create ImGUI Renderer!");

        self.im_context = Some(imgui);
        self.im_sdl2 = Some(platform);
        self.im_render = Some(renderer);
    }

    unsafe fn init_triangle_pipeline(&mut self){
        let triangle_vert_shader =
            load_shader_module(String::from("shaders/vert.spv"), self.get_device())
                .expect("Unable to load shader module!");

        let triangle_frag_shader =
            load_shader_module(String::from("shaders/frag.spv"), self.get_device())
                .expect("Unable to load shader module!");

        let push_constant = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<vk_types::GPUDrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();
        let push_constants = [push_constant];
        let pipeline_layout_info = pipeline_layout_create_info()
            .push_constant_ranges(&push_constants)
            .build();

        self.triangle_pipeline_layout = self.get_device().create_pipeline_layout(
            &pipeline_layout_info, None
        )
            .expect("Unable to create triangle pipeline layout!");

        let mut pipeline_builder = PipelineBuilder::new();
        pipeline_builder.pipeline_layout = self.triangle_pipeline_layout;
        pipeline_builder.set_shaders(triangle_vert_shader, triangle_frag_shader);
        pipeline_builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        pipeline_builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE);
        pipeline_builder.set_multisampling_none();
        pipeline_builder.enable_blending_additive();
        pipeline_builder.enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL);
        pipeline_builder.rasterizer.line_width = 1.0;

        pipeline_builder.set_color_attachment_format(self.draw_image.image_format);
        pipeline_builder.set_depth_format(self.depth_image.image_format);

        self.triangle_pipeline = pipeline_builder.build_pipeline(self.get_device());

        self.main_deletion_queue.push_pipeline(self.triangle_pipeline);
        self.main_deletion_queue.push_pipeline_layout(self.triangle_pipeline_layout);

    }

    pub unsafe fn create_buffer(&mut self, alloc_size: vk::DeviceSize, usage:
    vk::BufferUsageFlags, memory_usage:
    vk_mem::MemoryUsage) -> AllocatedBuffer{
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(alloc_size)
            .usage(usage);

        let mut vma_alloc_info = vk_mem::AllocationCreateInfo::default();
        vma_alloc_info.usage = memory_usage;
        vma_alloc_info.flags = vk_mem::AllocationCreateFlags::MAPPED
            .bitor(
                vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
            );

        let mut buffer = AllocatedBuffer::default();

        let alloc = self.allocator.as_ref().unwrap()
            .create_buffer(&buffer_info,&vma_alloc_info)
            .expect("Unable to create buffer!");

        buffer.buffer = alloc.0;
        buffer.allocation = Some(alloc.1);

        buffer
    }

    pub unsafe fn upload_mesh(&mut self, indices: &[i32], vertices: &[vk_types::Vertex])
    -> vk_types::GPUMeshBuffers{
        let vertex_buffer_size = vertices.len() * std::mem::size_of::<vk_types::Vertex>();
        let index_buffer_size = indices.len() * std::mem::size_of::<i32>();

        let mut new_surface = vk_types::GPUMeshBuffers::default();

        new_surface.vertex_buffer = self.create_buffer(
            vk::DeviceSize::from(vertex_buffer_size as u32), vk::BufferUsageFlags::STORAGE_BUFFER.bitor(
                vk::BufferUsageFlags::TRANSFER_DST.bitor(
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                )
            ), vk_mem::MemoryUsage::AutoPreferDevice
        );

        let device_address_info = vk::BufferDeviceAddressInfo::builder()
            .buffer(new_surface.vertex_buffer.buffer);
        new_surface.vertex_buffer_address = self.get_device().get_buffer_device_address(
            &device_address_info
        );

        new_surface.index_buffer = self.create_buffer(
            vk::DeviceSize::from(index_buffer_size as u32),
            vk::BufferUsageFlags::INDEX_BUFFER.bitor(
                vk::BufferUsageFlags::TRANSFER_DST
            ), vk_mem::MemoryUsage::AutoPreferDevice
        );

        let mut staging = self.create_buffer(
            vk::DeviceSize::from(
                (vertex_buffer_size + index_buffer_size) as u32
            ),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost
        );

        let data = self.allocator.as_ref().unwrap().map_memory(
            staging.allocation.as_mut().unwrap())
            .expect("Unable to map memory!");

        data.copy_from(
            vertices.as_ptr().cast(),
            vertex_buffer_size
        );

        data.offset(vertex_buffer_size as isize)
            .copy_from(indices.as_ptr().cast(),
                       index_buffer_size);

        self.immediate_submit(&|device: &ash::Device,
                                cmd: vk::CommandBuffer|{
            let vertex_copy = vk::BufferCopy::builder()
                .dst_offset(0)
                .src_offset(0)
                .size(vk::DeviceSize::from(vertex_buffer_size as u32))
                .build();

            let vertex_region = [vertex_copy];

            device.cmd_copy_buffer(cmd, staging.buffer,
            new_surface.vertex_buffer.buffer, &vertex_region);

            let index_copy = vk::BufferCopy::builder()
                .dst_offset(0)
                .src_offset(
                    vk::DeviceSize::from(vertex_buffer_size as u32)
                )
                .size(vk::DeviceSize::from(index_buffer_size as u32))
                .build();

            let index_region = [index_copy];

            device.cmd_copy_buffer(cmd, staging.buffer,
                                   new_surface.index_buffer.buffer, &index_region);
        });

        self.allocator.as_ref().unwrap().unmap_memory(
            staging.allocation.as_mut().unwrap()
        );

        self.delete_buffer(&mut staging);

        new_surface
    }

    pub unsafe fn delete_buffer(&self, buffer: &mut AllocatedBuffer){
        self.allocator.as_ref().unwrap()
            .destroy_buffer(buffer.buffer, buffer.allocation.as_mut().unwrap());
    }

    pub unsafe fn delete_allocation(&self, buffer: vk::Buffer, allocation: &vk_mem::Allocation){
        self.get_device().destroy_buffer(buffer, None);
        self.allocator.as_ref().unwrap().free_memory(allocation);
    }

    pub unsafe fn cleanup(&mut self){
        if self.is_initialized{
            self.get_device().device_wait_idle()
                .expect("Unable to wait idle");

            self.main_deletion_queue.flush(self);

            MeshAsset::delete_mesh(&mut self.test_mesh.clone(), self);

            for i in 0..FRAME_OVERLAP{
                self.get_device().destroy_command_pool(self.frames[i].command_pool, None);

                self.get_device().destroy_fence(self.frames[i].render_fence, None);
                self.get_device().destroy_semaphore(self.frames[i].render_semaphore, None);
            }

            self.destroy_swapchain();

            self.surface_dev.as_ref().unwrap().destroy_surface(self.surface, None);
            self.get_device().destroy_device(None);
            self.debug_utils.as_ref().unwrap().destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.as_ref().unwrap().destroy_instance(None);
        }
    }
}