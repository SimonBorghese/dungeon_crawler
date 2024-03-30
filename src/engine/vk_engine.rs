use sdl2::event::WindowEvent;
use ash_bootstrap;
use ash;
use ash::vk;
use ash::vk::Handle;

const USE_VALIDATION_LAYERS: bool = true;

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
    pub swapchain_image_format: ash::vk::Format,
    pub swapchain_images: Vec<ash::vk::Image>,
    pub swapchain_image_views: Vec<ash::vk::ImageView>,
    pub swapchain_extent: ash::vk::Extent2D,
}

impl VulkanEngine{
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
            swapchain_image_format: std::default::Default::default(),
            swapchain_images: vec![],
            swapchain_image_views: vec![],
            swapchain_extent: std::default::Default::default(),
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
                    ash_bootstrap::ValidationLayers::Require
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
            .require_features(&features)
            .for_surface(self.surface)
            .require_extension(ash::extensions::khr::Swapchain::name().as_ptr())
            .build(self.instance.as_ref().unwrap(),
            self.surface_dev.as_ref().unwrap(), &builder.2)
            .expect("Unable to build device!");

        self.device = Some(selector.0);
        self.chosen_gpu = selector.1.physical_device();

        println!("Successfully made device: {}", selector.1.device_name());

        Ok(())
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

        let mut swapchain_builder = ash_bootstrap::SwapchainOptions::new();
        swapchain_builder.format_preference(&[ash::vk::SurfaceFormatKHR::builder()
                                   .format(self.swapchain_image_format)
                                   .color_space(ash::vk::ColorSpaceKHR::SRGB_NONLINEAR)
                                    .build()]);
        swapchain_builder.present_mode_preference(&[ash::vk::PresentModeKHR::FIFO]);
        swapchain_builder.usage(ash::vk::ImageUsageFlags::TRANSFER_DST);
        let swapchain_builder2 =
            ash_bootstrap::Swapchain::new(swapchain_builder.clone(), self.surface, self.chosen_gpu,
            self.device.as_ref().unwrap(),
                                          ash::extensions::khr::Swapchain
                                          ::new(self.instance.as_ref().unwrap(),
                                          self.device.as_ref().unwrap()), self.swapchain_extent);



        self.swapchain = Some(swapchain_builder2);
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

            image_views.push(self.device.as_ref().unwrap().create_image_view(&info, None)
                .expect("Unable to make image view"));
        }
        image_views
    }

    unsafe fn destroy_swapchain(&mut self){
        self.swapchain.as_mut().unwrap().destroy(self.device.as_ref().unwrap());

        for image in &self.swapchain_image_views{
            self.device.as_ref().unwrap().destroy_image_view(*image, None);
        }
    }

    unsafe fn init_commands(&mut self){

    }

    unsafe fn init_sync_structures(&mut self){

    }

    pub unsafe fn cleanup(&mut self){
        if (self.is_initialized){
            self.destroy_swapchain();

            self.surface_dev.as_ref().unwrap().destroy_surface(self.surface, None);
            self.device.as_ref().unwrap().destroy_device(None);
            self.debug_utils.as_ref().unwrap().destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.as_ref().unwrap().destroy_instance(None);
        }
    }
}