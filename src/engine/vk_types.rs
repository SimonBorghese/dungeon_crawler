#![allow(dead_code)]
use ash::{Device, vk};
use ash;
use vk_mem;
use glm;
use vk_mem::Allocator;

pub trait VulkanObject {
    unsafe fn free(&self, device: &ash::Device, allocator: &vk_mem::Allocator);
}

#[derive(Default)]
pub struct AllocatedImage{
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Option<vk_mem::Allocation>,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
    pub allocator: Option<vk_mem::Allocator>,
}

impl VulkanObject for AllocatedImage{
    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
        device.destroy_image_view(self.image_view, None);

        allocator.destroy_image(self.image, self.allocation.as_ref().unwrap());
    }
}


#[derive(Default)]
pub struct AllocatedBuffer{
    pub buffer: vk::Buffer,
    pub allocation: Option<vk_mem::Allocation>,
    pub info: Option<vk_mem::AllocationInfo>
}

impl VulkanObject for AllocatedBuffer{
    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
        allocator.destroy_buffer(self.buffer, self.allocation.as_ref().unwrap());
    }
}


impl Clone for AllocatedBuffer{
    fn clone(&self) -> Self {
        Self{
            buffer: self.buffer,
            allocation: self.allocation.clone(),
            info: self.info.clone()
        }
    }
}

#[repr(C)]
pub struct Vertex{
    pub position: glm::Vec3,
    pub uv_x: f32,
    pub normal: glm::Vec3,
    pub uv_y: f32,
    pub color: glm::Vec4
}

#[derive(Default, Clone)]
pub struct GPUMeshBuffers{
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}

impl VulkanObject for GPUMeshBuffers{
    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
        self.index_buffer.free(device, allocator);
        self.vertex_buffer.free(device, allocator);
    }
}

pub struct GPUDrawPushConstants{
    pub world_matrix: glm::Matrix4<f32>,
    pub vertex_buffer: vk::DeviceAddress,
}