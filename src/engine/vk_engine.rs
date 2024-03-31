use sdl2::event::WindowEvent;
use ash_bootstrap;
use ash;
use ash::vk;
use ash::vk::{CommandPoolCreateFlags, Handle};
use ash_bootstrap::QueueFamilyCriteria;
use super::vk_initializers::*;
use super::vk_image;

const USE_VALIDATION_LAYERS: bool = true;

#[derive(Default, Copy, Clone)]
pub struct FrameData {
    command_pool: ash::vk::CommandPool,
    command_buffer: ash::vk::CommandBuffer,
    render_semaphore: ash::vk::Semaphore,
    render_fence: ash::vk::Fence,
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
    pub window_extent: ash::vk::Extent2D,

    pub instance: Option<ash::Instance>,
    pub debug_messenger: ash::vk::DebugUtilsMessengerEXT,
    pub chosen_gpu: ash::vk::PhysicalDevice,
    pub device: Option<ash::Device>,
    pub surface: ash::vk::SurfaceKHR,
    pub surface_dev: Option<ash::extensions::khr::Surface>,
    pub debug_utils: Option<ash::extensions::ext::DebugUtils>,
    
    pub swapchain: Option<ash_bootstrap::swapchain::Swapchain>,
    pub swapchain_dev: Option<ash::extensions::khr::Swapchain>,
    pub swapchain_image_format: ash::vk::Format,
    pub swapchain_images: Vec<ash::vk::Image>,
    pub swapchain_image_views: Vec<ash::vk::ImageView>,
    pub swapchain_extent: ash::vk::Extent2D,
    
    pub frames: [FrameData; FRAME_OVERLAP],
    pub graphics_queue: ash::vk::Queue,
    pub graphics_queue_family: u32
}

impl VulkanEngine{
    pub fn get_current_frame(&self) -> &FrameData{
        &self.frames[self.frame_number as usize % FRAME_OVERLAP]
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
            window_extent: ash::vk::Extent2D::builder().width(width).height(height).build(),
            instance: None,
            debug_messenger: std::default::Default::default(),
            debug_utils: None,
            chosen_gpu: std::default::Default::default(),
            device: std::default::Default::default(),
            surface: std::default::Default::default(),
            surface_dev: None,
            swapchain: None,
            swapchain_dev: None,
            swapchain_image_format: std::default::Default::default(),
            swapchain_images: vec![],
            swapchain_image_views: vec![],
            swapchain_extent: std::default::Default::default(),
            frames: [FrameData::default(); 2],
            graphics_queue: Default::default(),
            graphics_queue_family: 0,
        }
    }

    pub fn init(&mut self){

        unsafe {
            self.init_vulkan()
                .expect("Unable to initialize Vulkan");
            self.init_swapchain();
            self.init_commands();
            self.init_sync_structures();
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

    unsafe fn draw(&mut self){
        self.get_device().wait_for_fences(&[self.get_current_frame().render_fence],
                                          true, 1000000000)
            .expect("Unable to wait for fence");

        self.get_device().reset_fences(&[self.get_current_frame().render_fence])
            .expect("Unable to reset fence!");

        let swapchain_image = self.swapchain.as_mut().unwrap()
            .acquire(self.device.as_mut().unwrap(), self.surface_dev.as_ref().unwrap(),
                     1000000000, false)
            .expect("Unable to acquire swapchain image");

        let cmd = self.get_current_frame().command_buffer;

        self.get_device().reset_command_buffer(cmd, Default::default())
            .expect("Unable to reset command buffer!");

        let cmd_begin_info = command_buffer_begin_info(
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        );

        self.get_device().begin_command_buffer(cmd, &cmd_begin_info)
            .expect("Unable to begin command buffer!");

        vk_image::transition_image(self.get_device(),
        cmd, self.swapchain_images[swapchain_image.frame_index],
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        let mut clear_value;

        let flash = glm::abs(glm::cos(self.frame_number as f32 / 120.0));
        clear_value = vk::ClearColorValue::default();
        clear_value.float32[2] = flash;
        clear_value.float32[3] = 1.0;

        let clear_range = image_subresource_range(
            vk::ImageAspectFlags::COLOR
        );

        self.get_device().cmd_clear_color_image(cmd,
        self.swapchain_images[swapchain_image.frame_index],
        vk::ImageLayout::GENERAL, &clear_value, &[clear_range.build()]);

        vk_image::transition_image(self.get_device(),
        cmd, self.swapchain_images[swapchain_image.frame_index],
        vk::ImageLayout::GENERAL, vk::ImageLayout::PRESENT_SRC_KHR);

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
            self.get_current_frame().render_semaphore
        ).build();

        let submit = submit_info(&[cmd_info],
                                 &[signal_info],
                                 &[wait_info])
            .build();

        self.get_device().queue_submit2(self.graphics_queue, &[submit],
        self.get_current_frame().render_fence)
            .expect("Unable to submit queue");

        /*
        let swapchain = [self.swapchain.as_ref().unwrap().handle()];
        let render_semaphore = [self.get_current_frame().render_semaphore];
        let image_index = [swapchain_image.image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchain)
            .wait_semaphores(&render_semaphore)
            .image_indices(&image_index);

        self.swapchain_dev.as_ref().unwrap().queue_present(self.graphics_queue, &present_info)
            .expect("Unable t opresent");

         */

        let semaphore = self.get_current_frame().render_semaphore;
        self.swapchain.as_mut().unwrap().queue_present(self.graphics_queue,
                                                       semaphore, swapchain_image.image_index)
            .expect("Unable to present image!");

        self.get_device().reset_fences(&[swapchain_image.complete])
            .expect("Unable to reset fence!");
        self.frame_number += 1;
        println!("Frame Number: {}", self.frame_number);
    }


    extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        type_: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let data = unsafe { *data };
        let message = unsafe { std::ffi::CStr::from_ptr(data.p_message) }.to_string_lossy();

        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            println!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            println!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            println!("({:?}) {}", type_, message);
        } else {
            println!("({:?}) {}", type_, message);
        }

        vk::FALSE
    }
    unsafe fn init_vulkan(&mut self) -> Result<(), ash_bootstrap::InstanceCreationError>{
        let callback: ash::vk::PFN_vkDebugUtilsMessengerCallbackEXT = Some(Self::debug_callback);
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

        self.surface = ash::vk::SurfaceKHR::from_raw(surface_handle);
        self.surface_dev =
            Some(ash::extensions::khr::Surface::new(&self.entry, self.instance.as_ref().unwrap()));

        let vulkan_13_features =
            ash::vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true)
            .build();

        let mut vulkan_12_features =
            ash::vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .build();

        vulkan_12_features.p_next = &vulkan_13_features as *const _ as *mut _;

        let mut features = ash::vk::PhysicalDeviceFeatures2::builder()
            .build();
        features.p_next = &vulkan_12_features as *const _ as *mut _;

        let selector = ash_bootstrap::DeviceBuilder::new()
            .require_version(1,3)
            .set_required_features_12(vulkan_12_features)
            .set_required_features_13(vulkan_13_features)
            .for_surface(self.surface)
            .require_extension(ash::extensions::khr::Swapchain::name().as_ptr())
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
        self.swapchain_image_format = ash::vk::Format::B8G8R8A8_UNORM;
        self.swapchain_extent = ash::vk::Extent2D::builder()
            .width(width)
            .height(height)
            .build();

        let swapchain_builder = ash_bootstrap::SwapchainOptions::new()
            .format_preference(&[ash::vk::SurfaceFormatKHR::builder()
                                   .format(self.swapchain_image_format)
                                   .color_space(ash::vk::ColorSpaceKHR::SRGB_NONLINEAR)
                                    .build()])
            .frames_in_flight(3)
            .present_mode_preference(&[ash::vk::PresentModeKHR::FIFO])
            .usage(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
             | ash::vk::ImageUsageFlags::TRANSFER_DST);

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
    }

    unsafe fn create_image_views(&self, images: &Vec<ash::vk::Image>)
        -> Vec<ash::vk::ImageView>{
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

        let mut image_views: Vec<ash::vk::ImageView> = vec![];
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
    }

    pub unsafe fn cleanup(&mut self){
        if self.is_initialized{
            self.get_device().device_wait_idle()
                .expect("Unable to wait idle");

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