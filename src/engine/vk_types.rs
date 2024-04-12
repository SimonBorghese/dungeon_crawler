use ash::vk;
use ash;
use vk_mem;
use sdl2;
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

#[repr(C)]
pub struct Vertex{
    pub position: glm::Vec3,
    pub uv_x: f32,
    pub normal: glm::Vec3,
    pub uv_y: f32,
    pub color: glm::Vec4
}

#[derive(Default)]
pub struct GPUMeshBuffers{
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}

#[derive(Default)]
pub struct GPUDrawPushConstants{
    pub world_matrix: glm::Mat4,
    pub vertex_buffer: vk::DeviceAddress,
}