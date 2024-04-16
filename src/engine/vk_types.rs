#![allow(dead_code)]
use ash::vk;
use ash;
use vk_mem;
use glm;

#[derive(Default)]
pub struct AllocatedImage{
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Option<vk_mem::Allocation>,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
    pub allocator: Option<vk_mem::Allocator>,
}

#[derive(Default)]
pub struct AllocatedBuffer{
    pub buffer: vk::Buffer,
    pub allocation: Option<vk_mem::Allocation>,
    pub info: Option<vk_mem::AllocationInfo>
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

pub struct GPUDrawPushConstants{
    pub world_matrix: glm::Matrix4<f32>,
    pub vertex_buffer: vk::DeviceAddress,
}