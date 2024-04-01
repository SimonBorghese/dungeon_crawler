use ash::vk;
use ash;
use vk_mem;
use sdl2;

#[derive(Default)]
pub struct AllocatedImage{
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Option<vk_mem::Allocation>,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
}