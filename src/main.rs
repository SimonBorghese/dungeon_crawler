#![allow(
dead_code,
unused_variables,
clippy::too_many_arguments,
clippy::unnecessary_wraps
)]

use sdl2;
use sdl2::event::Event;
use anyhow::{anyhow, Result};
use log::*;
use vulkanalia::Version;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::InstanceCreateInfo;
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use thiserror::Error;

use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

const VALIDATION_ENABLED: bool =
    cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

fn main() -> Result<()> {
    pretty_env_logger::init();
    let sdl = sdl2::init()
        .expect("Unable to initialize SDL 2");

    // Window
    let video = sdl.video()
        .expect("Unable to get video");
    let window = video.window("Vulkan RS", 800, 600)
        .vulkan()
        .position_centered()
        .build()
        .expect("Couldn't build window!");
    let mut event = sdl.event_pump()
        .expect("Unable to get event pump");

    // App

    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;
    
    for event in event.poll_iter(){
        match event{
            Event::Quit { .. } => {
                unsafe{
                    app.destroy();
                }
            }

            _ => {
                unsafe{
                    app.render(&window)
                } .unwrap()
            }
        }
    }
    Ok(())
}

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &sdl2::video::Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)
            .expect("Couldn't load vulkan library");
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let instance = create_instance(window, &entry)?;
        let mut data = AppData::default();
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        Ok(Self {entry, instance, data, device})
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &sdl2::video::Window) -> Result<()> {
        Ok(())
    }

    /// Destroys our Vulkan app.
    unsafe fn destroy(&mut self) {
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);
        self.instance.destroy_instance(None);
    }
}

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData
) -> Result<Device>{
    let indices = QueueFamilyIndices
    ::get(instance, data, data.physical_device)
        .expect("Unable to get queue families");

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED{
        vec![VALIDATION_LAYER.as_ptr()]
    } else{
        vec![]
    };

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    let features = vk::PhysicalDeviceFeatures::builder();

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)
        .expect("Couldn't create device");

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

unsafe fn create_instance(window: &sdl2::video::Window, entry: &Entry) -> Result<Instance>{
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Rust\0")
        .application_version(vk::make_version(1,0,0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1,0,0))
        .api_version(vk::make_version(1,0,0));

    let extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER){
        return Err(anyhow!("Validation layer requested but not supported"));
    }

    let layers = if VALIDATION_ENABLED{
        vec![VALIDATION_LAYER.as_ptr()]
    } else{
        Vec::new()
    };

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_extension_names(&extensions)
        .enabled_layer_names(&layers);
    info.enabled_layer_count = 1;

    Ok(entry.create_instance(&info, None)?)
}

unsafe fn create_swapchain(
    window: &sdl2::video::Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData
) -> Result<()>{
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;

    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count{
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];

    let image_sharing_mode = if indices.graphics != indices.present{
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else{
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
         .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null()).build();

    data.swapchain = device.create_swapchain_khr(&info, None)?;

    Ok(())
}

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()>{
    for physical_device in instance.enumerate_physical_devices()?{
        let properties = instance
            .get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device){
            warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
        } else{
            println!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            return Ok(())
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice
) -> Result<()>{
    let properties = instance.get_physical_device_properties(physical_device);
    let features = instance.get_physical_device_features(physical_device);
    // Only 1 GPU in this test system

    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;
    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty(){
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice
) -> Result<()> {

    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains((e))){
        Ok(())
    } else{
        Err(anyhow!(SuitabilityError("Missing Required device extensions")))
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices{
    graphics: u32,
    present: u32
}

impl QueueFamilyIndices{
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice
    ) -> Result<Self>{
        let mut present = None;
        let properties = instance
            .get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        for (index, properties) in properties.iter().enumerate(){
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some (present)) = (graphics, present){
            Ok(Self { graphics, present })
        } else{
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport{
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>
}

impl SwapchainSupport{
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice
    ) -> Result<Self>{
        Ok(Self{
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?
        })
    }
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR{
    formats
        .iter()
        .cloned()
        .find(|f|
        f.format == vk::Format::B8G8R8A8_SRGB
        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR{
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(
    window: &sdl2::video::Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D{
    if capabilities.current_extent.width != u32::MAX{
        capabilities.current_extent
    } else{
        let size = window.size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.0
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.1,
            ))
            .build()
    }
}